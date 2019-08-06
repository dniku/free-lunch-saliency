import tarfile
import tempfile
from pathlib import Path

import cv2 as cv
import gym
import imageio
from gym.utils import seeding
from tqdm.auto import tqdm


def read_frames(path, frame_idx_watermark=False, progress=False):
    with tarfile.open(path, 'r:bz2') as tar:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            tar.extractall(tmpdir)
            members = [member for member in tar.getmembers() if member.isfile()]
            members.sort(key=lambda member: int(Path(member.name).stem.split('_')[-1]))

            it = members
            if progress:
                it = tqdm(it, postfix='reading frames')

            for member in it:
                if not member.isfile():
                    continue
                assert member.name.endswith('.png')
                with (tmpdir / member.name).open('rb') as fp:
                    img = imageio.imread(fp)
                    if frame_idx_watermark:
                        # Filenames start from 1, but for consistency idx starts from 0
                        idx = int(Path(member.name).stem.split('_')[-1]) - 1
                        cv.putText(
                            img,
                            str(idx),
                            (10 + 35 * int(idx % 4 >= 2), 130 + 40 * (idx % 2)),
                            cv.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(255, 255, 255),
                            lineType=cv.LINE_AA,
                        )
                    yield img


class StreamingImageEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, tar_path, base_env_name):
        self.tar_path = tar_path
        self.base_env = gym.make(base_env_name)
        self.action_space = self.base_env.action_space
        self.observation_space = self.base_env.observation_space
        self.gen = None
        self._last_frame = None
        self._next_frame = None
        self.np_random = None
        self.seed()
        self.ale = self.base_env.unwrapped.ale

    def reset(self):
        if self.gen is not None:
            self.gen.close()
        self.gen = read_frames(self.tar_path)
        assert not self._advance_frame()
        assert not self._advance_frame()
        return self._last_frame

    def step(self, _):
        done = self._advance_frame()
        return self._last_frame, 0, done, {}

    def _advance_frame(self):
        self._last_frame = self._next_frame
        try:
            self._next_frame = next(self.gen)
            done = False
        except StopIteration:
            self._next_frame = None
            done = True
        return done

    def render(self, mode='human'):
        return self._last_frame

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_action_meanings(self):
        return self.base_env.unwrapped.get_action_meanings()

    def close(self):
        self.gen = None
