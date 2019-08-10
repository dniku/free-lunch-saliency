import argparse
from typing import List, Tuple
import csv
import pickle
from collections import OrderedDict
from pathlib import Path

import gym
import numpy as np
import pandas as pd
import streaming_image_env
from gym.envs.registration import registry as env_registry

from metrics import nss, kldiv, auc_shuffled

import run_baselines
from utils import maybe_tqdm, assert_equal
from visualization import upscale_smap

# Prevent PyCharm from removing this import
assert hasattr(streaming_image_env, 'StreamingImageEnv')


class AtariHead:
    def __init__(self, dataset_dir: Path):
        self.dataset_dir = dataset_dir
        self.meta = self._read_meta()

    def _read_meta(self):
        meta = pd.read_csv(self.dataset_dir / 'meta_data.csv').dropna(axis=0)
        meta['TrialNumber'] = meta['TrialNumber'].astype(int)
        meta['NumberOfFrames'] = meta['NumberOfFrames'].astype(int)

        data = {
            int(path.name.split('_')[0]): path.name[:-len('.tar.bz2')]
            for path in self.dataset_dir.iterdir()
            if path.name.endswith('.tar.bz2')
        }
        meta['frames'] = meta['TrialNumber'].map(lambda tn: data[tn] + '.tar.bz2')
        meta['gazes'] = meta['TrialNumber'].map(lambda tn: data[tn] + '.txt')

        meta['Game'] = meta['Game'].str.replace('Mspacman', 'MsPacman')

        return meta

    def read_gazes(self, trial_number: int):
        run_gazes = self.get_run(trial_number).gazes
        cols = ['frame_id', 'episode_id', 'score', 'duration', 'unclipped_reward', 'action', 'gaze_positions']

        records = []
        with (self.dataset_dir / run_gazes).open() as fp:
            csv_reader = csv.reader(fp)
            next(csv_reader)  # header
            for row in csv_reader:
                row = [(None if item == 'null' else item) for item in row]
                values = row[:len(cols) - 1]
                gazes = row[len(cols) - 1:]
                if gazes == [None]:
                    gazes = []
                else:
                    gazes = [float(v) for v in gazes]
                    gazes = list(zip(gazes[::2], gazes[1::2]))
                values.append(gazes)
                d = OrderedDict(zip(cols, values))
                for field in ['duration', 'unclipped_reward', 'action']:
                    if d[field] is not None:
                        d[field] = int(d[field])
                records.append(d)

        df_gazes = pd.DataFrame.from_records(records)

        df_gazes['gaze_num'] = df_gazes['gaze_positions'].map(len)

        df_gazes['run_name'] = df_gazes['frame_id'].map(lambda s: s[:s.rindex('_')])
        df_gazes['frame_idx'] = df_gazes['frame_id'].map(lambda s: int(s[s.rindex('_') + 1:]))

        return df_gazes

    def get_run(self, trial_number: int):
        matches = self.meta[self.meta['TrialNumber'] == trial_number]
        assert len(matches) == 1
        return matches.iloc[0]

    def get_env_name(self, trial_number):
        env_name = f'StreamingImageEnvNoFrameskip{trial_number}-v0'

        if env_name not in env_registry.env_specs:
            run = self.get_run(trial_number)
            gym.envs.register(
                id=env_name,
                entry_point='streaming_image_env:StreamingImageEnv',
                kwargs={'tar_path': self.dataset_dir / run['frames'], 'base_env_name': run['Game'] + 'NoFrameskip-v4'}
            )

        return env_name

    def game_trials(self, game: str):
        return self.meta[self.meta['Game'] == game]['TrialNumber']


def make_fixation_map(fixations: List[Tuple[float, float]]):
    def inc_in_bounds(a, i, j):
        if 0 <= i < a.shape[0] and 0 <= j < a.shape[1]:
            a[i, j] += 1
    rh, rw = 210, 160
    ph, pw = 84, 84
    raw_fixation_map = np.zeros((rh, rw), dtype=np.int64)
    prc_fixation_map = np.zeros((ph, pw), dtype=np.int64)
    for x, y in fixations:
        inc_in_bounds(raw_fixation_map, round(y), round(x))
        inc_in_bounds(prc_fixation_map, round(y / rh * ph), round(x / rw * pw))
    return raw_fixation_map, prc_fixation_map


def auc_shuffled_many(saliency_maps, fixation_maps, random_state=None, progress=False):
    gen = np.random.RandomState(seed=random_state)
    result = []

    tqdm = maybe_tqdm(progress)
    it = tqdm(enumerate(zip(saliency_maps, fixation_maps)), total=len(saliency_maps), postfix='evaluating sAUC')

    for i, (smap, fmap) in it:
        if np.allclose(fmap, 0) or not np.isfinite(smap).all():
            v_sauc = None
        else:
            omap_indices = gen.randint(len(fixation_maps) - 1, size=(10,))
            omap_indices[omap_indices >= i] += 1
            omap = np.sum([fixation_maps[j] for j in omap_indices], axis=0)

            v_sauc = auc_shuffled(smap, fmap, omap, random_state=random_state)
        result.append(v_sauc)

    return result


def idx_for_framestack(i):
    return [
        i - j
        for j in reversed(range(14))
        if (0 <= (j % 4) <= 1) and (i - j >= 0)
    ]


def render_fixation_video(trial_number: int, atari_head: AtariHead, progress=False):
    run = atari_head.get_run(trial_number)
    df_gazes = atari_head.read_gazes(trial_number)

    frame_stream = streaming_image_env.read_frames(atari_head.dataset_dir / run['frames'])

    tqdm = maybe_tqdm(progress)
    it = tqdm(zip(frame_stream, df_gazes['gaze_positions']), total=run['NumberOfFrames'], postfix='writing video')

    for frame, frame_gazes in it:
        raw_fixation_map, prc_fixation_map = make_fixation_map([frame_gazes])

        m = raw_fixation_map.max()
        if not np.allclose(m, 0):
            raw_fixation_map = raw_fixation_map / m

        frame = (frame / 255 + raw_fixation_map[..., np.newaxis])[np.newaxis, ...] / 2
        yield frame


def model_trial_maps(
        model,
        network_name: str,
        trial_number: int,
        atari_head: AtariHead,
        eval_seed: int,
        return_raw=True,
        return_prc=False,
        progress=False):
    env_name = atari_head.get_env_name(trial_number)

    _, raw_observations, prc_observations, saliency_maps = run_baselines.evaluate_model(
        model,
        network_name=network_name,
        env_name=env_name,
        num_env=1,
        seed=eval_seed,
        frame_stack_size=4,
        # a lower bound that always seems to work is run['NumberOfFrames'] // 4 + 2
        max_eplen=None,
        noop_reset=False,
        fire_reset=False,
        eval_dir=None,
        return_raw=return_raw,
        return_prc=return_prc,
        progress=progress,
    )

    saliency_maps = [smap[0] for smap in saliency_maps]  # 1st dimension is for batch

    df_gazes = atari_head.read_gazes(trial_number)

    last_frame = atari_head.get_run(trial_number)['NumberOfFrames'] - 1
    if last_frame % 4 == 0:
        last_frame -= 4

    raw_fixation_maps, prc_fixation_maps = zip(*[
        make_fixation_map([
            (y, x)
            for frame_gazes in df_gazes['gaze_positions'][df_gazes['frame_idx'].isin(idx_for_framestack(i))]
            for (y, x) in frame_gazes
        ])
        for i in range(0, last_frame + 1, 4)
    ])

    if return_raw:
        if return_prc:
            assert_equal(len(raw_observations), len(prc_observations), len(saliency_maps), len(raw_fixation_maps))
        else:
            assert_equal(len(raw_observations), len(saliency_maps), len(raw_fixation_maps))
    elif return_prc:
        assert_equal(len(prc_observations), len(saliency_maps), len(raw_fixation_maps))
        assert_equal(prc_observations[0].shape[1:-1], saliency_maps[0].shape, raw_fixation_maps[0].shape)
    else:
        assert_equal(len(saliency_maps), len(raw_fixation_maps))
        assert_equal(saliency_maps[0].shape, raw_fixation_maps[0].shape)

    return raw_observations, prc_observations, saliency_maps, raw_fixation_maps, prc_fixation_maps


def model_trial_metrics(
        model,
        network_name: str,
        trial_number: int,
        atari_head: AtariHead,
        eval_seed: int,
        processed_obs=False,
        progress=False):
    raw_observations, prc_observations, saliency_maps, raw_fixation_maps, prc_fixation_maps = model_trial_maps(
        model,
        network_name=network_name,
        trial_number=trial_number,
        atari_head=atari_head,
        eval_seed=eval_seed,
        return_prc=processed_obs,
        progress=progress,
    )

    run_metrics = {
        metric: []
        for metric in ['nss', 'kldiv']
    }

    if not processed_obs:
        saliency_maps = [upscale_smap(smap[np.newaxis, ...])[0] for smap in saliency_maps]

    prg = maybe_tqdm(progress)
    for smap, fmap in prg(zip(saliency_maps, raw_fixation_maps), total=len(raw_observations), postfix='computing NSS & KL-div'):
        if np.allclose(fmap, 0) or not np.isfinite(smap).all():
            v_nss = None
            v_kldiv = None
        else:
            v_nss = nss(smap, fmap)

            if np.allclose(smap, 0):
                v_kldiv = None
            else:
                v_kldiv = kldiv(smap, fmap, fixation_sigma=2 if processed_obs else 5)

        run_metrics['nss'].append(v_nss)
        run_metrics['kldiv'].append(v_kldiv)

    run_metrics['sauc'] = auc_shuffled_many(saliency_maps, raw_fixation_maps, random_state=42)

    run_metrics = {key: np.array(value, dtype=np.float64) for key, value in run_metrics.items()}

    return run_metrics


def model_metrics(
        experiment_dir: Path,
        atari_head: AtariHead,
        eval_seed: int,
        processed_obs: bool,
        progress=False):
    records = []

    d, model = run_baselines.load_model(
        experiment_dir,
        eval_seed,
    )

    game = d['env_name'][:-len('NoFrameskip-v4')]
    network = d['network']

    tqdm = maybe_tqdm(progress)
    for trial_number in tqdm(atari_head.game_trials(game), postfix=f'runs: {game} {network}'):
        metrics = model_trial_metrics(
            model,
            d['network'],
            trial_number,
            atari_head,
            eval_seed=eval_seed,
            processed_obs=processed_obs,
        )
        records.append(OrderedDict([
            ('trial_number', trial_number),
            ('metrics', metrics),
        ]))

    return records


def main(experiment_dir: Path,
         atari_head_dir: Path,
         eval_seed: int,
         processed_obs: bool,
         progress: bool,
         output_dir: Path):
    output_path = output_dir / 'saliency.pkl'
    assert not output_path.exists()

    atari_head = AtariHead(atari_head_dir)
    records = model_metrics(
        experiment_dir,
        atari_head,
        eval_seed,
        processed_obs,
        progress,
    )

    with output_path.open('wb') as fp:
        pickle.dump(records, fp)


if __name__ == '__main__':
    import warnings

    warnings.simplefilter('error')
    np.seterr(divide='raise', over='raise', under='ignore', invalid='raise')

    parser = argparse.ArgumentParser(description='Compute saliency metrics on trained models using Atari-HEAD dataset')
    parser.add_argument('--experiment-dir', type=Path, help='Path to directory with model.pkl and config.json', required=True)
    parser.add_argument('--atari-head-dir', type=Path, help='Directory with Atari-HEAD dataset', required=True)
    parser.add_argument('--eval-seed', type=int, help='Seed to pass to env initializers (should not affect anything)', default=1000)
    parser.add_argument('--processed-obs', action='store_true', help='Compute metrics on preprocessed obs instead of raw ones')
    parser.add_argument('--progress', action='store_true', help='Use tqdm for reporting progress')
    parser.add_argument('--output-dir', type=Path, help='Directory to write pickled records into', required=True)
    args = parser.parse_args()

    main(
        args.experiment_dir,
        args.atari_head_dir,
        args.eval_seed,
        args.processed_obs,
        args.progress,
        args.output_dir,
    )
