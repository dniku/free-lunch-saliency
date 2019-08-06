from pathlib import Path

import cv2
import numpy as np
from tqdm.auto import tqdm


def gpu_is_available_tensorflow():
    from baselines.common.tf_util import get_available_gpus
    return bool(get_available_gpus())


def maybe_tqdm(progress=False):
    if progress:
        return tqdm
    else:
        return lambda it, *args, **kwargs: it


def assert_equal(*args):
    assert all(args[0] == args[i] for i in range(1, len(args))), args


class VideoWriter:
    def __init__(self, path: Path, fps: float = 30.0, fourcc: str = 'FFV1'):
        assert len(fourcc) == 4
        self.path = path
        self.fps = float(fps)
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.out = None
        self.frame_size = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.out is not None:
            self.out.release()

    def write(self, frame):
        assert len(frame.shape) == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        frame_size = (frame.shape[1], frame.shape[0])
        if self.out is None:
            self.path.parent.mkdir(exist_ok=True, parents=True)
            self.frame_size = frame_size
            args = [str(self.path), self.fourcc, self.fps, self.frame_size]
            self.out = cv2.VideoWriter(*args)
        else:
            assert self.frame_size == frame_size, f"Wrong frame size: should be {self.frame_size}, got {frame_size}"
        self.out.write(frame)
