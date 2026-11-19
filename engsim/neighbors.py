"""O(N) 3D Linked-Cell lists and Verlet neighbor lists with displacement-triggered rebuilds."""

from __future__ import annotations

import numpy as np

from .potentials import minimum_image, wrap_positions


def build_cell_list(pos: np.ndarray, box_L: float, rc: float) -> tuple[dict[tuple[int, int, int], list[int]], int]:
    """Assign particles to 3D linked cells of edge >= rc."""
    n_cells = max(1, int(np.floor(box_L / rc)))
    wrapped = wrap_positions(pos, box_L)
    cell_size = box_L / n_cells
    cell_idx = np.floor(wrapped / cell_size).astype(int) % n_cells
    cells: dict[tuple[int, int, int], list[int]] = {}
    for p_idx, (cx, cy, cz) in enumerate(cell_idx):
        key = (int(cx), int(cy), int(cz))
        cells.setdefault(key, []).append(p_idx)
    return cells, n_cells


def lj_forces_cell_list(
    pos: np.ndarray,
    box_L: float,
    rc: float = 2.5,
    shifted: bool = True,
) -> tuple[np.ndarray, float, float]:
    """Compute exact Lennard-Jones forces, potential energy, and virial in O(N) using 3D linked-cell lists."""
    n_atoms = pos.shape[0]
    cells, n_cells = build_cell_list(pos, box_L, rc)
    if n_cells < 3:
        # Box too small for distinct 27-cell stencil; fall back to exact all-pairs
        from .potentials import lj_forces_all_pairs
        return lj_forces_all_pairs(pos, box_L, rc=rc, shifted=shifted)

    # 13 forward half-shell neighbor offsets in 3D (plus self-cell)
    half_offsets = [
        (1, 0, 0), (1, 1, 0), (1, -1, 0), (0, 1, 0),
        (0, 0, 1), (1, 0, 1), (-1, 0, 1), (0, 1, 1), (0, -1, 1),
        (1, 1, 1), (1, -1, 1), (-1, 1, 1), (-1, -1, 1),
    ]

    rc2 = rc * rc
    inv_rc6 = (1.0 / rc2) ** 3
    u_shift = 4.0 * (inv_rc6 * inv_rc6 - inv_rc6) if shifted else 0.0

    forces = np.zeros_like(pos)
    u_tot = 0.0
    virial = 0.0

    for (cx, cy, cz), p_list in cells.items():
        p_arr = np.array(p_list, dtype=int)
        # 1. Intra-cell pairs
        if len(p_arr) > 1:
            sub_pos = pos[p_arr]
            dr = minimum_image(sub_pos[:, None, :] - sub_pos[None, :, :], box_L)
            r2 = np.sum(dr * dr, axis=2)
            iu, ju = np.triu_indices(len(p_arr), k=1)
            r2_p = r2[iu, ju]
            mask = r2_p < rc2
            if np.any(mask):
                ia, ja = p_arr[iu[mask]], p_arr[ju[mask]]
                dr_a = dr[iu[mask], ju[mask]]
                r2_a = r2_p[mask]
                inv_r2 = 1.0 / r2_a
                inv_r6 = inv_r2**3
                inv_r12 = inv_r6**2
                u_tot += float(np.sum(4.0 * (inv_r12 - inv_r6) - u_shift))
                f_sc = 24.0 * inv_r2 * (2.0 * inv_r12 - inv_r6)
                f_v = f_sc[:, None] * dr_a
                np.add.at(forces, ia, f_v)
                np.add.at(forces, ja, -f_v)
                virial += float(np.sum(f_sc * r2_a))

        # 2. Inter-cell half-shell pairs
        for ox, oy, oz in half_offsets:
            nb_key = ((cx + ox) % n_cells, (cy + oy) % n_cells, (cz + oz) % n_cells)
            q_list = cells.get(nb_key)
            if not q_list:
                continue
            q_arr = np.array(q_list, dtype=int)
            dr = minimum_image(pos[p_arr][:, None, :] - pos[q_arr][None, :, :], box_L)
            r2 = np.sum(dr * dr, axis=2)
            pi_loc, qj_loc = np.where(r2 < rc2)
            if len(pi_loc) > 0:
                ia, ja = p_arr[pi_loc], q_arr[qj_loc]
                dr_a = dr[pi_loc, qj_loc]
                r2_a = r2[pi_loc, qj_loc]
                inv_r2 = 1.0 / r2_a
                inv_r6 = inv_r2**3
                inv_r12 = inv_r6**2
                u_tot += float(np.sum(4.0 * (inv_r12 - inv_r6) - u_shift))
                f_sc = 24.0 * inv_r2 * (2.0 * inv_r12 - inv_r6)
                f_v = f_sc[:, None] * dr_a
                np.add.at(forces, ia, f_v)
                np.add.at(forces, ja, -f_v)
                virial += float(np.sum(f_sc * r2_a))

    return forces, u_tot, virial


class VerletNeighborList:
    """Verlet neighbor list with automatic skin-displacement rebuild trigger."""

    def __init__(self, rc: float = 2.5, skin: float = 0.35) -> None:
        self.rc = float(rc)
        self.skin = float(skin)
        self.rv = self.rc + self.skin
        self.pairs_i = np.empty(0, dtype=int)
        self.pairs_j = np.empty(0, dtype=int)
        self.ref_pos: np.ndarray | None = None
        self.n_rebuilds = 0

    def needs_rebuild(self, pos: np.ndarray, box_L: float) -> bool:
        if self.ref_pos is None:
            return True
        disp = minimum_image(pos - self.ref_pos, box_L)
        max_disp = float(np.max(np.linalg.norm(disp, axis=1)))
        return (2.0 * max_disp) > self.skin

    def build(self, pos: np.ndarray, box_L: float) -> None:
        n_atoms = pos.shape[0]
        dr = minimum_image(pos[:, None, :] - pos[None, :, :], box_L)
        r2 = np.sum(dr * dr, axis=2)
        iu, ju = np.triu_indices(n_atoms, k=1)
        mask = r2[iu, ju] < (self.rv * self.rv)
        self.pairs_i = iu[mask]
        self.pairs_j = ju[mask]
        self.ref_pos = pos.copy()
        self.n_rebuilds += 1

    def forces(self, pos: np.ndarray, box_L: float, shifted: bool = True) -> tuple[np.ndarray, float, float]:
        if self.needs_rebuild(pos, box_L):
            self.build(pos, box_L)
        if self.pairs_i.size == 0:
            return np.zeros_like(pos), 0.0, 0.0

        dr = minimum_image(pos[self.pairs_i] - pos[self.pairs_j], box_L)
        r2 = np.sum(dr * dr, axis=1)
        rc2 = self.rc * self.rc
        mask = r2 < rc2
        if not np.any(mask):
            return np.zeros_like(pos), 0.0, 0.0

        ia = self.pairs_i[mask]
        ja = self.pairs_j[mask]
        dr_a = dr[mask]
        r2_a = r2[mask]
        inv_r2 = 1.0 / r2_a
        inv_r6 = inv_r2**3
        inv_r12 = inv_r6**2
        inv_rc6 = (1.0 / rc2) ** 3
        u_shift = 4.0 * (inv_rc6 * inv_rc6 - inv_rc6) if shifted else 0.0

        u_tot = float(np.sum(4.0 * (inv_r12 - inv_r6) - u_shift))
        f_sc = 24.0 * inv_r2 * (2.0 * inv_r12 - inv_r6)
        f_v = f_sc[:, None] * dr_a

        forces = np.zeros_like(pos)
        np.add.at(forces, ia, f_v)
        np.add.at(forces, ja, -f_v)
        virial = float(np.sum(f_sc * r2_a))
        return forces, u_tot, virial
