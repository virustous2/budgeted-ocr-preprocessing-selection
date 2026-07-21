#!/usr/bin/env python3
"""Shared preprocessing candidate sampling utilities."""

from __future__ import annotations

from itertools import product
from typing import Dict, List

import numpy as np
import scipy
from scipy.stats import qmc

DISCRETE_PARAM_VALUES = {
    # ksize=1 is an identity 1x1 blur in OpenCV, so it is excluded.
    'gaussian_ksize': [0, 3, 5, 7],
    'bilateral_d': list(range(10)),
}

CONTINUOUS_PARAM_BOUNDS = {
    'clahe_clip': (0.0, 4.0),
    'gamma': (0.5, 1.5),
    'sharpen': (0.0, 2.0),
}

# Keep numeric bounds for stable parameter ordering and downstream compatibility.
# The executed discrete support for gaussian_ksize/bilateral_d is defined in
# DISCRETE_PARAM_VALUES and enforced by LHSSampler/validate_executed_params.
PREPROCESS_BOUNDS = {
    'gaussian_ksize': (0, 7),
    'bilateral_d': (0, 9),
    **CONTINUOUS_PARAM_BOUNDS,
}

PARAM_NAMES = list(PREPROCESS_BOUNDS.keys())


def _bin_index(value: float, n_bins: int) -> int:
    if n_bins <= 0:
        raise ValueError(f'n_bins must be positive, got {n_bins}')
    if not np.isfinite(value):
        raise ValueError(f'Cannot bin non-finite value: {value}')

    # LatinHypercube samples should lie in [0, 1), but clamp defensively so a
    # boundary value of 1.0 still maps to the last executed level.
    clipped = float(np.clip(value, 0.0, np.nextafter(1.0, 0.0)))
    return int(np.floor(clipped * n_bins))


def validate_executed_params(params: Dict[str, float]) -> None:
    gaussian = int(params['gaussian_ksize'])
    bilateral = int(params['bilateral_d'])

    if gaussian not in DISCRETE_PARAM_VALUES['gaussian_ksize']:
        raise ValueError(f"Invalid gaussian_ksize={gaussian}; expected one of "
                         f"{DISCRETE_PARAM_VALUES['gaussian_ksize']}")
    if bilateral not in DISCRETE_PARAM_VALUES['bilateral_d']:
        raise ValueError(f"Invalid bilateral_d={bilateral}; expected one of "
                         f"{DISCRETE_PARAM_VALUES['bilateral_d']}")


class LHSSampler:
    """Sample executed preprocessing params by binning a stratified 5D unit cube."""

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.param_names = PARAM_NAMES
        self.n_params = len(self.param_names)

    def sample(self, n_samples: int) -> List[Dict[str, float]]:
        sampler = qmc.LatinHypercube(
            d=self.n_params,
            scramble=True,
            strength=1,
            seed=self.seed,
        )
        unit_samples = sampler.random(n=n_samples)
        param_samples = []

        gaussian_levels = DISCRETE_PARAM_VALUES['gaussian_ksize']
        bilateral_levels = DISCRETE_PARAM_VALUES['bilateral_d']

        for sample in unit_samples:
            params = {
                'gaussian_ksize': gaussian_levels[_bin_index(sample[0], len(gaussian_levels))],
                'bilateral_d': bilateral_levels[_bin_index(sample[1], len(bilateral_levels))],
            }

            for i, name in enumerate(CONTINUOUS_PARAM_BOUNDS, start=2):
                low, high = CONTINUOUS_PARAM_BOUNDS[name]
                params[name] = float(low + sample[i] * (high - low))

            validate_executed_params(params)
            param_samples.append(params)

        return param_samples


def _discrete_joint_coverage(rows: np.ndarray) -> Dict[str, object]:
    discrete_names = list(DISCRETE_PARAM_VALUES.keys())
    if not discrete_names:
        return {
            'params': [],
            'unique_combinations': 0,
            'total_possible_combinations': 0,
            'has_duplicate_combinations': False,
            'missing_combinations': [],
            'combination_counts': {},
        }

    n_discrete = len(discrete_names)
    discrete_rows = rows[:, :n_discrete].astype(int)
    unique_rows, counts = np.unique(discrete_rows, axis=0, return_counts=True)
    observed = {
        tuple(int(v) for v in row.tolist()): int(count)
        for row, count in zip(unique_rows, counts)
    }
    all_combinations = list(product(
        *[DISCRETE_PARAM_VALUES[name] for name in discrete_names]
    ))

    def format_key(values) -> str:
        return '|'.join(str(int(v)) for v in values)

    return {
        'params': discrete_names,
        'unique_combinations': len(observed),
        'total_possible_combinations': len(all_combinations),
        'has_duplicate_combinations': len(observed) != len(discrete_rows),
        'missing_combinations': [
            format_key(combo) for combo in all_combinations
            if tuple(combo) not in observed
        ],
        'combination_counts': {
            format_key(combo): count
            for combo, count in sorted(observed.items())
        },
    }


def sampler_validation_summary(param_samples: List[Dict[str, float]],
                               seed: int) -> Dict[str, object]:
    if not param_samples:
        return {
            'seed': seed,
            'scipy_version': scipy.__version__,
            'n_samples': 0,
            'unique_executed_rows': 0,
            'has_duplicate_rows': False,
            'discrete_counts': {},
            'discrete_joint_coverage': {},
            'continuous_strata': {},
        }

    for params in param_samples:
        validate_executed_params(params)

    n_samples = len(param_samples)
    rows = np.array([[params[name] for name in PARAM_NAMES] for params in param_samples], dtype=float)

    discrete_counts = {}
    for idx, name in enumerate(DISCRETE_PARAM_VALUES):
        unique, counts = np.unique(rows[:, idx].astype(int), return_counts=True)
        discrete_counts[name] = {str(int(u)): int(c) for u, c in zip(unique, counts)}

    continuous_strata = {}
    for idx, (name, bounds) in enumerate(CONTINUOUS_PARAM_BOUNDS.items(), start=2):
        low, high = bounds
        normalized = (rows[:, idx] - low) / (high - low)
        strata = np.floor(np.minimum(normalized, np.nextafter(1.0, 0.0)) * n_samples).astype(int)
        counts = np.bincount(strata, minlength=n_samples)
        continuous_strata[name] = {
            'all_strata_hit': bool(np.all(counts == 1)),
            'count_min': int(counts.min()),
            'count_max': int(counts.max()),
        }

    row_tuples = [tuple(row) for row in rows.tolist()]
    unique_rows = len(set(row_tuples))

    return {
        'seed': seed,
        'scipy_version': scipy.__version__,
        'n_samples': n_samples,
        'unique_executed_rows': unique_rows,
        'has_duplicate_rows': unique_rows != n_samples,
        'discrete_counts': discrete_counts,
        'discrete_joint_coverage': _discrete_joint_coverage(rows),
        'continuous_strata': continuous_strata,
    }
