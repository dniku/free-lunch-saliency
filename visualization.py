from typing import List

import numpy as np
import skimage.transform
import torch
import torch.nn.functional as F

from utils import maybe_tqdm, assert_equal


def render_attn_frame(attn: np.ndarray, receptive_field: int, stride: int, padding: int):
    attn = torch.Tensor(attn)
    b, h, w = attn.shape
    true_attn = torch.empty((
        b,
        receptive_field + (h - 1) * stride - 2 * padding,
        receptive_field + (w - 1) * stride - 2 * padding,
    ))
    flt = torch.ones((1, 1, receptive_field, receptive_field))
    for bb in range(b):
        true_attn[bb, :, :] = F.conv_transpose2d(
            attn[bb, :, :][None, None, ...],
            flt,
            stride=stride,
            padding=padding,
        )
    return true_attn.clamp(min=0).numpy()


def render_attention(
    attn: List[np.ndarray],
    obs_shape,
    receptive_field: int,
    stride: int,
    padding: int,
    progress=False) -> List[np.ndarray]:
    # attn: time * batch * height * width

    b, h, w, f = obs_shape
    ha = ((h + 2 * padding) - receptive_field) // stride + 1
    wa = ((w + 2 * padding) - receptive_field) // stride + 1

    assert attn[0].shape == (b, ha, wa), (obs_shape, attn[0].shape, (b, ha, wa))

    tqdm = maybe_tqdm(progress)
    true_attn = [
        render_attn_frame(frame, receptive_field=receptive_field, stride=stride, padding=padding)
        for frame in tqdm(attn, postfix='rendering attn')
    ]

    true_attn_max = max(frame.max() for frame in true_attn)
    if not np.allclose(true_attn_max, 0):
        for frame in true_attn:
            frame /= true_attn_max

    return true_attn


def render_frame(obs, smap, fmap=None, processed_obs=False):
    if processed_obs:
        obs = obs[..., -1]

    obs = obs.astype(np.float32) / 255

    if processed_obs:
        if fmap is None:
            fmap = np.zeros_like(obs)
        else:
            fmap_max = fmap.max()
            if fmap_max:
                fmap = fmap / fmap_max
        assert_equal(obs.shape, smap.shape, fmap.shape)
        frame = np.stack([fmap, smap, obs], axis=-1)
    else:
        b, h, w, c = obs.shape
        assert c == 3
        if fmap is None:
            assert_equal(obs.shape[:3], smap.shape)
            frame = 0.5 * (obs + smap[..., np.newaxis])
        else:
            assert_equal(obs.shape[:3], smap.shape, fmap.shape)
            frame = 0.5 * obs + 0.5 * smap[..., np.newaxis]
            fmap_max = fmap.max()
            if fmap_max:
                frame[..., 0] = np.maximum(frame[..., 0], fmap / fmap_max)
    return frame


def upscale_smap(smap):
    return np.stack([
        skimage.transform.resize(smap[bb, ...], (210, 160))
        for bb in range(smap.shape[0])
    ])


def render_perception(raw_observations: List[np.ndarray], prc_observations: List[np.ndarray],
                      saliency_maps: List[np.ndarray], raw_fixation_maps: List[np.ndarray] = None,
                      prc_fixation_maps: List[np.ndarray] = None):
    assert bool(raw_observations) or bool(prc_observations)
    assert len({len(raw_observations), len(prc_observations), len(saliency_maps)} - {0}) == 1

    def render_seq(observations, smaps, fmaps, processed_obs):
        if fmaps is None:
            for obs, smap in zip(observations, smaps):
                yield render_frame(obs, smap, processed_obs=processed_obs)
        else:
            for obs, smap, fmap in zip(observations, smaps, fmaps):
                yield render_frame(obs, smap, fmap, processed_obs=processed_obs)

    if raw_observations:
        num_env, rh, rw = raw_observations[0].shape[:3]

        upscaled_maps = (upscale_smap(smap) for smap in saliency_maps)

        if prc_observations:
            for frame_raw, frame_prc in zip(
                    render_seq(raw_observations, upscaled_maps, raw_fixation_maps, processed_obs=False),
                    render_seq(prc_observations, saliency_maps, prc_fixation_maps, processed_obs=True)):
                frame_prc = np.stack([skimage.transform.resize(frame_prc[bb, ...], (rw, rw)) for bb in range(num_env)])
                yield np.concatenate([frame_raw, frame_prc], axis=1)
        else:
            yield from render_seq(raw_observations, upscaled_maps, raw_fixation_maps, processed_obs=False)
    else:
        yield from render_seq(prc_observations, saliency_maps, prc_fixation_maps, processed_obs=True)
