import argparse
import json
import logging
import os
from pathlib import Path

import baselines.run

import models_baselines
from run_baselines import evaluate_model
from utils import gpu_is_available_tensorflow

with open('config.json', 'r') as fp:
    cfg = json.load(fp)


def main():
    assert gpu_is_available_tensorflow()

    parser = argparse.ArgumentParser(description='Test attention in CNNs with OpenAI Baselines')
    parser.add_argument(
        '--output-dir', required=True, type=Path,
        help='Path where all output will be stored')
    parser.add_argument(
        '--tensorboard-dir', required=True, type=Path,
        help='Path for Tensorboard logs')
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser()
    tensorboard_dir = args.tensorboard_dir.expanduser()

    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
    os.environ['OPENAI_LOGDIR'] = str(output_dir / 'logs')
    os.environ['TENSORBOARD_DIR'] = str(tensorboard_dir)

    network = cfg['model']['network']

    num_timesteps = cfg['train']['num_timesteps']
    num_env = cfg['train']['num_processes']

    save_video_interval = (num_timesteps // num_env) // (cfg['train']['num_videos'] - 1)

    # Besides ensuring correctness, this is also a workaround for PyCharm removing
    # the import of models_baselines, which registers networks.
    assert network in models_baselines.attention_visualization_params

    model = baselines.run.main([str(v) for v in [
        '--env', cfg['train']['env_name'],
        '--seed', cfg['train']['train_seed'],
        '--alg', cfg['train']['alg_name'],
        '--num_timesteps', num_timesteps,
        '--network', network,
        '--num_env', num_env,
        '--save_path', str(output_dir / 'model.pkl'),
        '--save_video_interval', save_video_interval,
        '--save_video_length', 1000,
    ]])

    eval_dir = output_dir / 'eval'
    eval_dir.mkdir(parents=True, exist_ok=True)

    evaluate_model(
        model,
        network_name=cfg['model']['network'],
        env_name=cfg['train']['env_name'],
        num_env=cfg['eval']['eval_num_envs'],
        evals_per_env=cfg['eval']['evals_per_env'],
        seed=cfg['eval']['eval_seed'],
        eval_dir=eval_dir,
        max_eplen=cfg['eval']['eval_max_eplen'],
        frame_stack_size=cfg['model']['frame_stack_size'],
    )


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)-15s] %(levelname)s: %(message)s'
    )
    main()
