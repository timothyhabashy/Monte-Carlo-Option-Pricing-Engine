"""Sobol normals and Brownian-bridge construction for QMC path generation."""

from __future__ import annotations

import numpy as np
from scipy.stats import norm
from scipy.stats.qmc import Sobol


def sobol_normals(dim: int, n: int, seed: int | None) -> np.ndarray:
    """Scrambled Sobol points mapped to N(0,1). Shape ``(dim, n)``."""

    if dim <= 0 or n <= 0:
        raise ValueError("dim and n must be positive")
    sampler = Sobol(d=dim, scramble=True, seed=seed)
    uniforms = np.clip(sampler.random(n), 1e-12, 1.0 - 1e-12)
    return np.asarray(norm.ppf(uniforms).T, dtype=float)


def brownian_bridge(z: np.ndarray, time_horizon: float) -> np.ndarray:
    """Build ``W`` at ``dt,...,T`` from i.i.d. normals via a Brownian bridge.

    ``z`` has shape ``(steps, n_paths)``. The first coordinate of ``z`` drives
    ``W_T``; remaining coordinates fill successive interval midpoints.
    """

    z = np.asarray(z, dtype=float)
    if z.ndim != 2:
        raise ValueError("z must have shape (steps, n_paths)")
    if time_horizon < 0:
        raise ValueError("time_horizon must be non-negative")
    steps, n_paths = z.shape
    if time_horizon == 0.0:
        return np.zeros((steps, n_paths), dtype=float)

    times = (np.arange(1, steps + 1, dtype=float) / steps) * time_horizon
    w = np.empty((steps, n_paths), dtype=float)
    w[-1] = np.sqrt(times[-1]) * z[0]
    used = 1
    queue: list[tuple[int, int]] = [(-1, steps - 1)]
    head = 0
    while used < steps:
        if head >= len(queue):
            raise RuntimeError("brownian bridge failed to fill the time grid")
        left, right = queue[head]
        head += 1
        if right - left <= 1:
            continue
        mid = (left + right) // 2
        if mid <= left:
            mid = left + 1
        if mid >= right:
            continue
        t_right = times[right]
        w_right = w[right]
        if left < 0:
            t_left = 0.0
            w_left = 0.0
        else:
            t_left = times[left]
            w_left = w[left]
        t_mid = times[mid]
        span = t_right - t_left
        mean = ((t_right - t_mid) * w_left + (t_mid - t_left) * w_right) / span
        var = (t_mid - t_left) * (t_right - t_mid) / span
        w[mid] = mean + np.sqrt(var) * z[used]
        used += 1
        queue.append((left, mid))
        queue.append((mid, right))
    return w
