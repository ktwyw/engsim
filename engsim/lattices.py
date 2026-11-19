"""Crystal lattice builders (SC, BCC, FCC) and Maxwell-Boltzmann velocity initialization."""

from __future__ import annotations

import numpy as np


def make_fcc_lattice(n_cells: int, rho: float) -> tuple[np.ndarray, float]:
    """Build a 3D Face-Centered Cubic (FCC) lattice with N = 4 * n_cells^3 particles at number density `rho`.

    Returns
    -------
    positions : np.ndarray, shape (N, 3)
        Particle coordinates inside [0, L)^3.
    box_L : float
        Cubic simulation box length L = (N / rho)^(1/3).
    """
    n_atoms = 4 * (n_cells**3)
    box_L = float((n_atoms / rho) ** (1.0 / 3.0))
    a = box_L / n_cells
    basis = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ],
        dtype=float,
    )
    coords = []
    for ix in range(n_cells):
        for iy in range(n_cells):
            for iz in range(n_cells):
                origin = np.array([ix, iy, iz], dtype=float)
                for b in basis:
                    coords.append((origin + b) * a)
    positions = np.array(coords, dtype=float) % box_L
    return positions, box_L


def make_sc_lattice(n_cells: int, rho: float) -> tuple[np.ndarray, float]:
    """Build a 3D Simple Cubic (SC) lattice with N = n_cells^3 particles."""
    n_atoms = n_cells**3
    box_L = float((n_atoms / rho) ** (1.0 / 3.0))
    a = box_L / n_cells
    grid = np.arange(n_cells, dtype=float) * a
    xx, yy, zz = np.meshgrid(grid, grid, grid, indexing="ij")
    positions = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return positions, box_L


def init_velocities(
    n_atoms: int,
    T_target: float,
    rng: np.random.Generator | None = None,
    dim: int = 3,
    exact_T: bool = True,
) -> np.ndarray:
    """Draw Maxwell-Boltzmann velocities with zero center-of-mass momentum."""
    gen = np.random.default_rng() if rng is None else rng
    vel = gen.normal(0.0, np.sqrt(T_target), size=(n_atoms, dim))
    vel -= np.mean(vel, axis=0, keepdims=True)
    if exact_T and n_atoms > 1:
        dof = dim * (n_atoms - 1)
        ke = 0.5 * float(np.sum(vel**2))
        T_curr = 2.0 * ke / dof
        if T_curr > 0.0:
            vel *= np.sqrt(T_target / T_curr)
    return vel
