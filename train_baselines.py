import argparse
import logging
import os
from pathlib import Path

import baselines.run

import models_baselines
from utils import gpu_is_available_tensorflow


def main():
    assert gpu_is_available_tensorflow()

    parser = argparse.ArgumentParser(description='Test attention in CNNs with OpenAI Baselines')
    parser.add_argument('--network', type=str, default='cnn_sparse_fls')
    parser.add_argument('--env-name', type=str, default='BreakoutNoFrameskip-v4')
    parser.add_argument('--train-seed', type=int, default=1)
    parser.add_argument('--num-timesteps', type=int, default=50000000)
    parser.add_argument('--output-dir', required=True, type=Path, help='Path where all output will be stored')
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser()

    os.environ['OPENAI_LOG_FORMAT'] = 'stdout,log,csv,tensorboard'
    os.environ['OPENAI_LOGDIR'] = str(output_dir / 'logs')
    os.environ['TENSORBOARD_DIR'] = str(output_dir / 'tb')

    # Besides ensuring correctness, this is also a workaround for PyCharm removing
    # the import of models_baselines, which registers networks.
    assert args.network in models_baselines.attention_visualization_params

    baselines.run.main([str(v) for v in [
        '--env', args.env_name,
        '--seed', args.train_seed,
        '--alg', 'ppo2',
        '--num_timesteps', args.num_timesteps,
        '--network', args.network,
        '--num_env', 8,
        '--save_path', str(output_dir / 'model.pkl'),
    ]])


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(asctime)-15s] %(levelname)s: %(message)s'
    )
    main()
