import argparse
import itertools
import json
import logging
import os
from contextlib import closing
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import baselines.run
import cv2
import numpy as np
from baselines.common.cmd_util import make_vec_env
from baselines.common.tile_images import tile_images
from baselines.common.vec_env import VecFrameStack
from baselines.common.vec_env.vec_video_recorder import VecVideoRecorder

import models_baselines
from utils import VideoWriter, maybe_tqdm
from visualization import render_attention, render_perception

from tensorflow.python.util import deprecation

deprecation._PRINT_DEPRECATION_WARNINGS = False

__all__ = ['load_model', 'evaluate_model']


def run_model(model, envs, evals_per_env: int, return_raw=True, return_prc=False, return_attn=True, progress=False):
    tqdm = maybe_tqdm(progress)

    num_envs = envs.venv.num_envs

    done_evals_per_env = np.zeros(num_envs, dtype=np.int64)
    num_done = 0
    eval_ep_stats = np.zeros(shape=(num_envs, evals_per_env, 3), dtype=np.float32)

    raw_observations = []
    prc_observations = []
    attention = []

    obs = envs.reset()

    # This approach ensures that we get num_envs truly random samples,
    # as opposed to an arbitrary number of samples that happened to finish first.
    # Of course, here we execute redundant steps on envs which have already ended,
    # but I don't see any easy way to avoid that.
    #
    # Note on terminology: for gym, "done" != "info contains reward and length".
    # Hence we run each env until it produces info with reward, length, and elapsed time.
    for _ in tqdm(itertools.count(start=0), postfix='playing'):
        actions, _, _, _, extra = model.step(obs)

        if return_raw:
            raw_observations.append(np.stack(envs.get_images()))
        if return_prc:
            # obs: num_env * height * width * frames
            prc_observations.append(obs.copy())

        attn = extra.copy()
        # Workaround for sloppy code in models_baselines.py
        if len(attn.shape) == 4 and attn.shape[-1] == 1:
            attn = np.squeeze(attn, axis=-1)

        if return_attn:
            # attention: num_env * height * width
            attention.append(attn)

        # Perform action, get reward and next observation
        obs, reward, done, infos = envs.step(actions)

        for i in range(num_envs):
            if done_evals_per_env[i] >= evals_per_env:
                if return_raw:
                    raw_observations[-1][i].fill(0)
                if return_prc:
                    prc_observations[-1][i].fill(0)
                if return_attn:
                    attention[-1][i].fill(0)

            if 'episode' in infos[i].keys():
                if done_evals_per_env[i] < evals_per_env:
                    eval_ep_stats[i, done_evals_per_env[i], :] = [infos[i]['episode'][key] for key in 'rlt']

                done_evals_per_env[i] += 1
                if done_evals_per_env[i] == evals_per_env:
                    num_done += 1
                    if num_done < num_envs:
                        remaining_percent = done_evals_per_env[
                                                done_evals_per_env < evals_per_env].mean() / evals_per_env * 100
                        logging.info(
                            f'{num_done}/{num_envs} envs done. Remaining envs are {remaining_percent:.2f}% done.')

        if num_done == num_envs:
            break

    assert (done_evals_per_env >= evals_per_env).all()
    rewards, lengths, elapsed_time = [eval_ep_stats.reshape(-1, 3)[:, i] for i in range(3)]

    eval_results = {
        'rewards': rewards.tolist(),
        'lengths': lengths.tolist(),
        'elapsed_time': elapsed_time.tolist(),
        'done_per_env': done_evals_per_env.tolist(),
    }

    return eval_results, raw_observations, prc_observations, attention


def make_envs(
    env_name,
    num_env,
    seed,
    max_eplen,
    frame_stack_size=4,
    noop_reset=True,
    fire_reset=True,
    eval_dir: Path = None,
    use_logger=True,
    video_recorder=False):
    eval_envs = make_vec_env(
        env_name, 'atari',
        num_env=num_env, seed=seed,
        max_episode_steps=max_eplen,
        noop_reset=noop_reset,
        use_logger=use_logger,
        wrapper_kwargs={'fire_reset': fire_reset},
    )
    eval_envs = VecFrameStack(eval_envs, frame_stack_size)
    if video_recorder:
        eval_envs = VecVideoRecorder(
            eval_envs,
            str(eval_dir / 'videos'),
            record_video_trigger=lambda _: True,
            video_length=max_eplen,
        )
    return eval_envs


def evaluate_model(
    model, network_name: str,
    env_name: str, num_env: int, seed: int,
    evals_per_env=1,
    eval_dir: Path = None,
    use_logger=True,
    max_eplen=None,
    frame_stack_size=4,
    noop_reset=True,
    fire_reset=True,
    return_raw=False,
    return_prc=False,
    video_recorder=False,
    progress=False):
    tqdm = maybe_tqdm(progress)

    if eval_dir is not None:
        eval_dir.mkdir(exist_ok=True, parents=True)

    logging.debug(f'Creating {num_env} instances of {env_name}...')
    with closing(make_envs(
        env_name=env_name,
        num_env=num_env,
        seed=seed,
        max_eplen=max_eplen,
        frame_stack_size=frame_stack_size,
        noop_reset=noop_reset,
        fire_reset=fire_reset,
        eval_dir=eval_dir,
        use_logger=use_logger,
        video_recorder=video_recorder,
    )) as eval_envs:
        logging.debug(f'Done creating envs. Running each for {evals_per_env} episodes, at most {max_eplen} each...')
        eval_results, raw_observations, prc_observations, attention = run_model(
            model,
            eval_envs,
            evals_per_env=evals_per_env,
            return_raw=return_raw,
            return_prc=return_prc,
            return_attn=return_raw or return_prc,
            progress=progress,
        )

    if eval_dir is not None:
        with (eval_dir / 'results.json').open('w') as fp:
            json.dump(eval_results, fp, indent=4)

    if return_raw or return_prc:
        logging.debug(f'Rendering {len(attention)} frames of attention...')
        rh, rw, rc = 210, 160, 3
        ph, pw, pc = 84, 84, frame_stack_size
        if return_raw:
            assert raw_observations[0].shape == (num_env, rh, rw, rc)
        if return_prc:
            assert prc_observations[0].shape == (num_env, ph, pw, pc)

        saliency_maps = render_attention(
            attention,
            (num_env, ph, pw, pc),
            **models_baselines.attention_visualization_params[network_name]
        )
        logging.debug('Done rendering.')

        if eval_dir is not None:
            with VideoWriter(eval_dir / 'perception.mkv') as writer:
                it = tqdm(
                    render_perception(raw_observations, prc_observations, saliency_maps),
                    postfix='writing video',
                    total=len(saliency_maps)
                )
                for frame in it:
                    frame = tile_images(frame)
                    frame = (frame * 255).astype(np.uint8)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.write(frame)
    else:
        saliency_maps = []

    return eval_results, raw_observations, prc_observations, saliency_maps


def load_model(experiment_path, seed):
    with (experiment_path / 'config.json').open() as fp:
        d = json.load(fp)
    return d, baselines.run.main([str(v) for v in [
        '--env', d['env_name'],
        '--seed', seed,
        '--alg', 'ppo2',
        '--num_timesteps', 0,
        '--network', d['network'],
        '--num_env', 1,
        '--load_path', str(experiment_path / 'model.pkl'),
    ]])


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)-15s] %(levelname)s: %(message)s'
    )

    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--experiment-dir', type=Path, help='Path to directory with model.pkl and config.json', required=True)
    parser.add_argument('--num-env', type=int, help='Number of evaluation envs', default=4)
    parser.add_argument('--evals-per-env', type=int, help='Number of full episodes to run on each env', default=1)
    parser.add_argument('--eval-seed', type=int, help='Seed to pass to env initializers', default=1000)
    parser.add_argument('--max-eplen', type=int, help='Maximum episode length', default=108000)
    parser.add_argument('--raw-obs', action='store_true', help='Save raw observations')
    parser.add_argument('--processed-obs', action='store_true', help='Save processed observations')
    parser.add_argument('--video-recorder', action='store_true', help='Use VideoRecorder from Baselines')
    parser.add_argument('--progress', action='store_true', help='Use tqdm for reporting progress')
    parser.add_argument('--output-dir', type=Path, help='Directory to write evaluation records into')
    args = parser.parse_args()

    d, model = load_model(
        args.experiment_dir,
        args.eval_seed,
    )

    logging.debug(f'Loaded model: network={d["network"]}, env_name={d["env_name"]}.')

    eval_dir = args.output_dir
    if eval_dir is not None:
        eval_dir.mkdir(exist_ok=True, parents=True)

    evaluate_model(
        model,
        network_name=d['network'],
        env_name=d['env_name'],
        num_env=args.num_env,
        evals_per_env=args.evals_per_env,
        seed=args.eval_seed,
        eval_dir=eval_dir,
        max_eplen=args.max_eplen,
        return_raw=args.raw_obs,
        return_prc=args.processed_obs,
        video_recorder=args.video_recorder,
        progress=args.progress,
    )
