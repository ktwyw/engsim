"""Pair potentials (LJ, LJTS, WCA), minimum-image convention, virial pressure, and tail corrections."""

from __future__ import annotations

import numpy as np


def wrap_positions(pos: np.ndarray, box_L: float) -> np.ndarray:
    """Wrap particle coordinates into the primary periodic box [0, L)^d."""
    return np.mod(pos, box_L)


def minimum_image(dr: np.ndarray, box_L: float) -> np.ndarray:
    """Apply the cubic periodic Minimum-Image Convention (MIC) to displacement vectors."""
    return dr - box_L * np.round(dr / box_L)


def lj_tail_corrections(rho: float, rc: float) -> tuple[float, float]:
    """Standard isotropic long-range tail corrections (u_tail per particle, P_tail) for unshifted LJ."""
    sr3 = (1.0 / rc) ** 3
    sr9 = sr3**3
    u_tail = (8.0 / 3.0) * np.pi * rho * (sr9 / 3.0 - sr3)
    p_tail = (16.0 / 3.0) * np.pi * (rho**2) * ((2.0 / 3.0) * sr9 - sr3)
    return float(u_tail), float(p_tail)


def lj_forces_all_pairs(
    pos: np.ndarray,
    box_L: float,
    rc: float = 2.5,
    shifted: bool = True,
    return_stress: bool = False,
) -> tuple:
    """Vectorized O(N^2) Lennard-Jones 12-6 forces, potential energy, and virial W = sum_{i<j} r_ij . f_ij.

    Parameters
    ----------
    pos : np.ndarray, shape (N, 3)
        Particle positions.
    box_L : float
        Cubic box length.
    rc : float
        Cutoff radius (default 2.5 sigma; use 2^(1/6) for WCA).
    shifted : bool
        If True (default), compute Truncated-and-Shifted LJ (LJTS) so u(rc) = 0.
    return_stress : bool
        If True, also return the 3x3 virial stress sum sum_{i<j} r_ij_alpha * f_ij_beta.
    """
    n_atoms = pos.shape[0]
    rc2 = rc * rc
    inv_rc6 = (1.0 / rc2) ** 3
    u_shift = 4.0 * (inv_rc6 * inv_rc6 - inv_rc6) if shifted else 0.0

    # Pairwise displacement matrix dr[i, j] = r_i - r_j
    dr = pos[:, None, :] - pos[None, :, :]
    dr = minimum_image(dr, box_L)
    r2 = np.sum(dr * dr, axis=2)

    # Upper-triangle mask for unique pairs i < j within cutoff
    iu, ju = np.triu_indices(n_atoms, k=1)
    r2_pairs = r2[iu, ju]
    mask = r2_pairs < rc2

    if not np.any(mask):
        zeros_f = np.zeros_like(pos)
        if return_stress:
            return zeros_f, 0.0, 0.0, np.zeros((3, 3))
        return zeros_f, 0.0, 0.0

    i_act = iu[mask]
    j_act = ju[mask]
    dr_act = dr[i_act, j_act]
    r2_act = r2_pairs[mask]

    inv_r2 = 1.0 / r2_act
    inv_r6 = inv_r2 * inv_r2 * inv_r2
    inv_r12 = inv_r6 * inv_r6

    u_pairs = 4.0 * (inv_r12 - inv_r6) - u_shift
    u_tot = float(np.sum(u_pairs))

    # f_ij = (24 / r^2) * (2 / r^12 - 1 / r^6) * dr_ij
    f_scalar = 24.0 * inv_r2 * (2.0 * inv_r12 - inv_r6)
    f_vec = f_scalar[:, None] * dr_act

    forces = np.zeros_like(pos)
    np.add.at(forces, i_act, f_vec)
    np.add.at(forces, j_act, -f_vec)

    virial = float(np.sum(f_scalar * r2_act))
    if return_stress:
        # stress_vir[a, b] = sum_{i<j} dr_a * f_b
        stress_vir = np.einsum("pa,pb->ab", dr_act, f_vec)
        return forces, u_tot, virial, stress_vir
    return forces, u_tot, virial


def lj_single_particle_energy(
    pos: np.ndarray,
    idx: int,
    trial_pos: np.ndarray,
    box_L: float,
    rc: float = 2.5,
    shifted: bool = True,
) -> float:
    """O(N) potential energy of particle `idx` placed at `trial_pos` interacting with all other N-1 particles."""
    dr = trial_pos[None, :] - pos
    dr = minimum_image(dr, box_L)
    r2 = np.sum(dr * dr, axis=1)
    r2[idx] = np.inf
    rc2 = rc * rc
    mask = r2 < rc2
    if not np.any(mask):
        return 0.0
    inv_r6 = (1.0 / r2[mask]) ** 3
    inv_rc6 = (1.0 / rc2) ** 3
    u_shift = 4.0 * (inv_rc6 * inv_rc6 - inv_rc6) if shifted else 0.0
    return float(np.sum(4.0 * (inv_r6 * inv_r6 - inv_r6) - u_shift))
