import cv2 as cv
import numpy as np
import scipy.stats
from sklearn.metrics import roc_auc_score


def nss(saliency_map, fixation_map):
    s_map_std = np.std(saliency_map)
    if np.isclose(s_map_std, 0):
        return None
    else:
        s_map_norm = (saliency_map - np.mean(saliency_map)) / s_map_std
        return np.average(s_map_norm, weights=fixation_map)


def kldiv(saliency_map, fixation_map, fixation_sigma):
    fixation_map = cv.GaussianBlur(
        fixation_map.astype(np.float64),
        (0, 0),  # determine filter size automatically
        sigmaX=fixation_sigma,
        sigmaY=fixation_sigma,
    )

    if np.isclose(saliency_map[fixation_map > 0], 0).any():
        return None

    return scipy.stats.entropy(fixation_map.ravel(), saliency_map.ravel())


def auc_shuffled(saliency_map, fixation_map, other_map, num_splits=100, random_state=None):
    assert saliency_map.shape == fixation_map.shape == other_map.shape
    assert np.all(np.isfinite(saliency_map))
    assert np.all(fixation_map >= 0)
    assert np.all(other_map >= 0)
    assert np.any(other_map > 0)

    smap = saliency_map.ravel()
    fmap = fixation_map.ravel()
    omap = other_map.ravel()

    s_min, s_max = np.min(smap), np.max(smap)
    if np.isclose(s_min, s_max):
        return None

    smap = (smap - s_min) / (s_max - s_min)

    num_pixels = smap.shape[0]
    num_fixations = np.sum(fmap)

    smap_true = []
    for v, cnt in zip(smap, fmap):
        smap_true.extend([v] * cnt)
    smap_true = np.array(smap_true)

    random_fixations = np.random.RandomState(seed=random_state).choice(
        np.arange(num_pixels),
        size=(num_splits, num_fixations),
        p=omap / omap.sum()
    )

    y_true = np.zeros(num_fixations * 2, dtype=np.uint8)
    y_true[num_fixations:] = 1

    aucs = []
    for s in range(num_splits):
        smap_rand = smap[random_fixations[s]]
        y_pred = np.concatenate([smap_rand, smap_true])

        aucs.append(roc_auc_score(y_true, y_pred))

    return np.mean(aucs)
