"""Exact expectations for uniformly sampled preprocessing shortlists."""

from __future__ import annotations

from math import comb
from typing import Iterable

import numpy as np


def expected_minimum_without_replacement(values: Iterable[float], k: int) -> float:
    """Return E[min(S)] for a uniform size-k subset sampled without replacement.

    If the sorted population is x[0] <= ... <= x[n-1], x[i] is the first
    selected value in C(n-i-1, k-1) of the C(n, k) possible subsets. This
    removes Monte Carlo noise and does not depend on candidate row ordering.
    """
    ordered = np.sort(np.asarray(list(values), dtype=float))
    if ordered.ndim != 1 or len(ordered) == 0:
        raise ValueError("values must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(ordered)):
        raise ValueError("values must contain only finite numbers")

    n = len(ordered)
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= {n}; got {k}")

    denominator = comb(n, k)
    expectation = sum(
        float(ordered[i]) * comb(n - i - 1, k - 1)
        for i in range(n - k + 1)
    ) / denominator
    return float(expectation)
