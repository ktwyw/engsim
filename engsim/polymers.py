"""Polymer simulation helpers: FENE-WCA Kremer-Grest chains, Rg/Re, and Casassa slit-pore partitioning."""

from __future__ import annotations

import numpy as np


def fene_bond_forces(pos: np.ndarray, k_fene: float = 30.0, R0: float = 1.5) -> tuple[np.ndarray, float]:
    """FENE bonded potential U = -0.5 * k * R0^2 * ln(1 - (r/R0)^2) along a linear chain of shape (N, 3)."""
    dr = pos[1:] - pos[:-1]
    r2 = np.sum(dr * dr, axis=1)
    R02 = R0 * R0
    ratio = np.clip(r2 / R02, 0.0, 0.999)
    u_fene = float(-0.5 * k_fene * R02 * np.sum(np.log(1.0 - ratio)))
    f_bond = (-k_fene / (1.0 - ratio))[:, None] * dr  # force on bead i+1 from bead i
    forces = np.zeros_like(pos)
    forces[:-1] -= f_bond
    forces[1:] += f_bond
    return forces, u_fene


def radius_of_gyration_sq(pos: np.ndarray) -> float:
    """Compute squared radius of gyration R_g^2 = (1/N) sum_i |r_i - r_cm|^2."""
    cm = np.mean(pos, axis=0, keepdims=True)
    diff = pos - cm
    return float(np.mean(np.sum(diff * diff, axis=1)))


def sample_freely_jointed_chains(
    n_chains: int,
    n_bonds: int,
    bond_len: float = 1.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate `n_chains` 3D ideal freely-jointed polymer chains with `n_bonds` steps (shape: n_chains, n_bonds+1, 3)."""
    gen = np.random.default_rng() if rng is None else rng
    # Uniform unit vectors on S^2
    u = gen.normal(0.0, 1.0, size=(n_chains, n_bonds, 3))
    u /= np.linalg.norm(u, axis=2, keepdims=True)
    steps = bond_len * u
    chains = np.zeros((n_chains, n_bonds + 1, 3), dtype=float)
    chains[:, 1:, :] = np.cumsum(steps, axis=1)
    return chains


def cabs_slit_partition_coefficient(chains: np.ndarray, slit_widths: np.ndarray) -> np.ndarray:
    """Compute exact equilibrium slit-pore partition coefficients K(H) for all slit widths H from ONE bulk sample.

    For a chain whose z-coordinate span is S_z = max(z) - min(z), the fraction of valid
    center-of-mass placements inside a hard slit of width H is max(0, 1 - S_z / H).
    """
    z_coords = chains[:, :, 2]
    spans_z = np.max(z_coords, axis=1) - np.min(z_coords, axis=1)
    H_arr = np.asarray(slit_widths, dtype=float)
    K_vals = np.array([float(np.mean(np.maximum(0.0, 1.0 - spans_z / H))) for H in H_arr])
    return K_vals
