"""Trajectory analysis: radial distribution function g(r), MSD, FFT autocorrelation, Green-Kubo, and block errors."""

from __future__ import annotations

import numpy as np

from .potentials import minimum_image


def compute_rdf(
    trajectory: np.ndarray,
    box_L: float,
    n_bins: int = 80,
    r_max: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the radial distribution function g(r) from a trajectory of shape (n_frames, N, 3)."""
    traj = np.asarray(trajectory, dtype=float)
    if traj.ndim == 2:
        traj = traj[None, :, :]
    n_frames, n_atoms, _ = traj.shape
    rmax = 0.5 * box_L if r_max is None else min(float(r_max), 0.5 * box_L)
    edges = np.linspace(0.0, rmax, n_bins + 1)
    hist = np.zeros(n_bins, dtype=float)
    iu, ju = np.triu_indices(n_atoms, k=1)

    for frame in traj:
        dr = minimum_image(frame[iu] - frame[ju], box_L)
        r = np.linalg.norm(dr, axis=1)
        counts, _ = np.histogram(r, bins=edges)
        hist += 2.0 * counts  # Account for both (i,j) and (j,i)

    rho = n_atoms / (box_L**3)
    shell_vol = (4.0 / 3.0) * np.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    norm = n_frames * n_atoms * rho * shell_vol
    g_r = hist / np.maximum(norm, 1e-16)
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    return r_centers, g_r


def compute_msd(unwrapped_traj: np.ndarray) -> np.ndarray:
    """Compute Mean-Squared Displacement <|r(t) - r(0)|^2> over multiple time origins from (n_frames, N, d)."""
    traj = np.asarray(unwrapped_traj, dtype=float)
    n_frames = traj.shape[0]
    msd = np.zeros(n_frames, dtype=float)
    for lag in range(1, n_frames):
        disp = traj[lag:] - traj[:-lag]
        msd[lag] = float(np.mean(np.sum(disp * disp, axis=-1)))
    return msd


def autocorrelation_fft(series: np.ndarray) -> np.ndarray:
    """Fast O(M log M) unbiased autocorrelation function <A(0) A(t)> via Wiener-Khinchin FFT."""
    x = np.asarray(series, dtype=float)
    n = x.shape[0]
    n_fft = 1 << (2 * n - 1).bit_length()
    fx = np.fft.rfft(x, n=n_fft, axis=0)
    acf_raw = np.fft.irfft(fx * np.conj(fx), n=n_fft, axis=0)[:n]
    counts = np.arange(n, 0, -1, dtype=float)
    if x.ndim == 1:
        return acf_raw / counts
    # Average over all particles and sum over spatial components for vector VACF <v(0) . v(t)>
    if x.ndim == 3:
        # shape (n_frames, N, 3) -> sum over axis=2, mean over axis=1
        return np.mean(np.sum(acf_raw, axis=2), axis=1) / counts
    return np.mean(acf_raw, axis=1) / counts


def green_kubo_integral(acf: np.ndarray, dt: float) -> np.ndarray:
    """Cumulative trapezoidal integral of an autocorrelation function."""
    y = np.asarray(acf, dtype=float)
    out = np.zeros_like(y)
    if len(y) > 1:
        out[1:] = np.cumsum(0.5 * (y[:-1] + y[1:]) * dt)
    return out


def flyvbjerg_petersen_block_errors(series: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Flyvbjerg-Petersen (1989) blocking transformation returning (block_sizes, std_errors)."""
    x = np.asarray(series, dtype=float).copy()
    sizes = []
    errors = []
    block_size = 1
    while x.size >= 4:
        n = x.size
        var = float(np.var(x, ddof=1))
        sizes.append(block_size)
        errors.append(float(np.sqrt(var / n)))
        half = n // 2
        x = 0.5 * (x[: 2 * half : 2] + x[1 : 2 * half : 2])
        block_size *= 2
    return np.array(sizes, dtype=float), np.array(errors, dtype=float)
