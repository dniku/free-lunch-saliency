# Free-Lunch Saliency via Attention in Atari Agents

Code for https://arxiv.org/abs/1908.02511.

Trained models are available [here](https://yadi.sk/d/vOk5JoP0208kqg). Directory structure is as follows:

```
<game>.<network>
└── <train_seed>
    ├── config.json
    ├── eval.pkl
    ├── events.out.tfevents.<timestamp>.<hostname>
    ├── model.pkl
    └── saliency.pkl  # only for cnn{,_daqn,_rsppo,_sparse_fls,_sparse_fls_pool,_dense_fls}
```

### Getting Docker image

```bash
docker pull dniku/fl-saliency
```

Alternatively, you can build it yourself:

```bash
cat Dockerfile | docker build -t fl-saliency -
```

### Training

```bash
mkdir /tmp/fl-saliency
docker run \
    -v $(pwd):/home/docker/fl-saliency \
    -v /tmp/fl-saliency:/home/docker/out \
    --gpus all --user=$(id -u):$(id -g) --rm -it \
    fl-saliency \
    python3 fl-saliency/train_baselines.py \
        --output-dir /home/docker/out \
        --network <NETWORK> \
        --env-name <ENV_NAME> \
        --train-seed <TRAIN_SEED> \
        --num-timesteps <NUM_TIMESTEPS>
```

Parameters:

 *  `NETWORK` can be one of the following
    ```
    cnn
    cnn_daqn
    cnn_rsppo
    cnn_rsppo_nopad
    cnn_sparse_fls
    cnn_sparse_fls_pool
    cnn_sparse_fls_norm
    cnn_sparse_fls_1x1
    cnn_sparse_fls_sp2
    cnn_sparse_fls_norelu
    cnn_sparse_fls_norelu_pool
    cnn_sparse_fls_h1
    cnn_sparse_fls_x3
    cnn_dense_fls
    cnn_dense_fls_norelu
    ```
 *  `ENV` can be any environment supported by OpenAI Gym. We used the following ones:
    ```
    BeamRiderNoFrameskip-v4
    BreakoutNoFrameskip-v4
    MsPacmanNoFrameskip-v4
    SpaceInvadersNoFrameskip-v4
    EnduroNoFrameskip-v4
    SeaquestNoFrameskip-v4
    ```
 *  `TRAIN_SEED` can be any integer. We used `1`, `9`, `17`, `25`, `33`.
 *  `NUM_TIMESTEPS` can be any integer ≥ `1024`. We used `50000000`. Use `1024` for testing.

Output will be saved in the directory specified by `--output-dir` in the following format:

```
logs
├── 0.N.monitor.csv  # logs for each of the 8 training environments
├── log.txt          # plain-text log with metrics (also printed to stdout)
└── progress.csv     # CSV log with metrics
tb
└── events.out.tfevents.<timestamp>.<hostname>  # Tensorboard log
model.pkl            # trained model in Baselines format
```

### Performance evaluation

Assuming that you downloaded the models to `~/data/fl-saliency`:

```bash
docker run \
    -v $(pwd):/home/docker/fl-saliency \
    -v ~/data/fl-saliency/Breakout.cnn_sparse_fls/01/:/home/docker/experiment \
    -v /tmp/fl-saliency:/home/docker/out \
    --gpus all --user=$(id -u):$(id -g) --rm -it \
    fl-saliency \
    python3 fl-saliency/run_baselines.py \
        --experiment-dir experiment \
        --output-dir out \
        --num-env <NUM_ENV> \
        --evals-per-env <EVALS_PER_ENV> \
        --progress
```

Parameters:

 *  `NUM_ENV`: how many environments to spawn in parallel. We used `16`. Use `1` or `2` for testing.
 *  `EVALS_PER_ENV`: how many times to evaluate in each environment. We used `512`. Use `1` for testing.
 *  `--raw-obs`: save a video called `perception.mkv` with raw observations and an attention overlay.
 *  `--processed-obs`: save a video called `perception.mkv` with preprocessed observations and an attention overlay.

If you pass both `--raw-obs` and `--processed-obs`, raw and preprocessed observations will be stacked vertically.

Evaluation results are saved in `results.json`. Example:

```json
{
    "rewards": [
        864.0
    ],
    "lengths": [
        6849.0
    ],
    "elapsed_time": [
        25.020986557006836
    ],
    "done_per_env": [
        1
    ]
}
```

`rewards`, `lengths`, and `elapsed_time` come from Baselines. Each entry corresponds to a finished episode. `lengths` contains number of steps while `elapsed_time` is the time since the environment was spawned, in seconds. `done_per_env` has an entry for each environment and counts how many episodes were finished there by the time evaluation is over.

### Saliency evaluation

```bash
docker run \
    -v $(pwd):/home/docker/fl-saliency \
    -v ~/data/fl-saliency/Breakout.cnn_sparse_fls/01:/home/docker/experiment \
    -v /tmp/fl-saliency:/home/docker/out \
    -v ~/data/atari_head:/home/docker/atari_head \
    --gpus all --user=$(id -u):$(id -g) --rm -it \
    fl-saliency \
    sh -c '\
    pip3 install --user -e fl-saliency/streaming-image-env &&
    python3 fl-saliency/benchmark_atari_head.py \
        --experiment-dir experiment \
        --atari-head-dir atari_head \
        --output-dir out \
        --progress'
```
