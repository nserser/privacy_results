from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterator

import numpy as np


def next_batch_probs(remaining_user: int, remaining_slots: int, batch_size: int) -> list[tuple[int, float]]:
    """Exact Hypergeom(remaining_slots, remaining_user, batch_size) probabilities."""
    r, S, b = int(remaining_user), int(remaining_slots), int(batch_size)
    if not (0 <= r <= S and 0 < b <= S):
        raise ValueError((r, S, b))
    denom = math.comb(S, b)
    lo = max(0, b - (S - r))
    hi = min(b, r)
    out = [(c, math.comb(r, c) * math.comb(S-r, b-c) / denom) for c in range(lo, hi+1)]
    z = sum(p for _, p in out)
    return [(c, p/z) for c, p in out]


def occupancy_probability(counts: tuple[int, ...] | list[int], batch_size: int, k: int | None = None) -> float:
    counts = tuple(int(c) for c in counts)
    M, b = len(counts), int(batch_size)
    kk = sum(counts) if k is None else int(k)
    if sum(counts) != kk or any(c < 0 or c > b for c in counts):
        return 0.0
    return math.prod(math.comb(b, c) for c in counts) / math.comb(M*b, kk)


def expected_collision_pairs(M: int, b: int, k: int) -> float:
    if k < 2 or b < 2:
        return 0.0
    return math.comb(k, 2) * (b - 1) / (M*b - 1)


def factorial_moment_batch(M: int, b: int, k: int) -> float:
    if k < 2 or b < 2:
        return 0.0
    return k*(k-1)*(b-1) / (M*(M*b-1))


def sample_occupancy(M: int, b: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Sequential exact sample from the multivariate-hypergeometric occupancy law."""
    if not 0 <= k <= M*b:
        raise ValueError("k must be in [0,M*b]")
    counts = np.zeros(M, dtype=np.int16)
    r, S = int(k), int(M*b)
    for j in range(M):
        if j == M-1:
            c = r
        else:
            c = int(rng.hypergeometric(r, S-r, b))
        counts[j] = c
        r -= c
        S -= b
    if r != 0:
        raise RuntimeError("occupancy sampler invariant failed")
    return counts


def sample_weighted_sum(M: int, b: int, k: int, weights: np.ndarray, rng: np.random.Generator) -> float:
    """O(M) exact occupancy sample; used only for optional Monte Carlo checks."""
    return float(np.dot(sample_occupancy(M, b, k, rng), np.asarray(weights, float)))


def enumerate_occupancies(M: int, b: int, k: int, max_patterns: int = 100000) -> Iterator[tuple[tuple[int, ...], float]]:
    """Enumerate occupancy count vectors and exact probabilities for small cases."""
    est = math.comb(M+k-1, k)
    if est > max_patterns:
        raise ValueError(f"occupancy pattern upper bound {est} exceeds max_patterns={max_patterns}")
    denom = math.comb(M*b, k)
    counts = [0]*M

    def rec(j: int, rem: int):
        if j == M:
            if rem == 0:
                c = tuple(counts)
                yield c, math.prod(math.comb(b, x) for x in c)/denom
            return
        lo = max(0, rem - b*(M-j-1))
        hi = min(b, rem)
        for x in range(lo, hi+1):
            counts[j] = x
            yield from rec(j+1, rem-x)
    yield from rec(0, k)
