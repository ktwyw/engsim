"""Monte Carlo methods: NVT-MC, NPT-MC, Grand Canonical MC, 2D Ising (Metropolis & Wolff), and Widom insertion."""

from __future__ import annotations

from collections import deque
import numpy as np

from .potentials import lj_forces_all_pairs, lj_single_particle_energy, wrap_positions


def run_nvt_mc(
    pos_init: np.ndarray,
    box_L: float,
    T: float,
    n_sweeps: int,
    max_disp: float = 0.15,
    rc: float = 2.5,
    shifted: bool = True,
    rng: np.random.Generator | None = None,
) -> dict[str, object]:
    """Run Metropolis NVT Monte Carlo for a Lennard-Jones fluid (`n_sweeps * N` trial particle moves)."""
    gen = np.random.default_rng() if rng is None else rng
    pos = pos_init.copy()
    n_atoms = pos.shape[0]
    beta = 1.0 / T
    _, u_curr, _ = lj_forces_all_pairs(pos, box_L, rc=rc, shifted=shifted)

    u_hist = np.zeros(n_sweeps, dtype=float)
    n_acc = 0
    n_tot = n_sweeps * n_atoms

    for sw in range(n_sweeps):
        for _ in range(n_atoms):
            idx = int(gen.integers(0, n_atoms))
            u_old = lj_single_particle_energy(pos, idx, pos[idx], box_L, rc=rc, shifted=shifted)
            trial = wrap_positions(pos[idx] + gen.uniform(-max_disp, max_disp, size=3), box_L)
            u_new = lj_single_particle_energy(pos, idx, trial, box_L, rc=rc, shifted=shifted)
            du = u_new - u_old
            if du <= 0.0 or gen.random() < np.exp(-beta * du):
                pos[idx] = trial
                u_curr += du
                n_acc += 1
        u_hist[sw] = u_curr / n_atoms

    return {
        "pos": pos,
        "u_per_atom": u_hist,
        "acceptance": float(n_acc / max(n_tot, 1)),
    }


def widom_excess_chemical_potential(
    pos: np.ndarray,
    box_L: float,
    T: float,
    n_insertions: int = 500,
    rc: float = 2.5,
    shifted: bool = True,
    rng: np.random.Generator | None = None,
) -> float:
    """Estimate excess chemical potential mu_ex = -T * ln <exp(-dU / T)> via Widom ghost-particle insertion."""
    gen = np.random.default_rng() if rng is None else rng
    beta = 1.0 / T
    n_atoms = pos.shape[0]
    boltz_sum = 0.0
    # Append a dummy slot for calling lj_single_particle_energy cleanly
    pos_ext = np.vstack([pos, np.zeros((1, 3))])
    for _ in range(n_insertions):
        trial = gen.uniform(0.0, box_L, size=3)
        du = lj_single_particle_energy(pos_ext, n_atoms, trial, box_L, rc=rc, shifted=shifted)
        if beta * du < 50.0:
            boltz_sum += float(np.exp(-beta * du))
    avg_boltz = max(boltz_sum / n_insertions, 1e-300)
    return float(-T * np.log(avg_boltz))


def ising_2d_energy_mag(spins: np.ndarray, J: float = 1.0) -> tuple[float, float]:
    """Compute energy per spin and absolute magnetization per spin for a 2D periodic Ising lattice."""
    nn_sum = np.roll(spins, 1, axis=0) + np.roll(spins, 1, axis=1)
    energy = -J * float(np.sum(spins * nn_sum))
    n_sites = spins.size
    return energy / n_sites, float(abs(np.mean(spins)))


def ising_2d_metropolis_sweep(spins: np.ndarray, beta: float, rng: np.random.Generator) -> np.ndarray:
    """Checkerboard-vectorized Metropolis sweep on a 2D periodic Ising lattice."""
    L = spins.shape[0]
    s = spins.copy()
    ii, jj = np.indices((L, L))
    for parity in (0, 1):
        mask = (ii + jj) % 2 == parity
        nn = np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1)
        dE = 2.0 * s * nn
        accept = (dE <= 0.0) | (rng.random((L, L)) < np.exp(-beta * np.maximum(dE, 0.0)))
        flip = mask & accept
        s[flip] *= -1
    return s


def ising_2d_wolff_step(spins: np.ndarray, beta: float, rng: np.random.Generator) -> tuple[np.ndarray, int]:
    """Single Wolff (1989) cluster flip on a 2D periodic Ising lattice."""
    L = spins.shape[0]
    s = spins.copy()
    p_add = 1.0 - float(np.exp(-2.0 * beta))
    i0, j0 = int(rng.integers(0, L)), int(rng.integers(0, L))
    spin0 = int(s[i0, j0])

    visited = np.zeros((L, L), dtype=bool)
    visited[i0, j0] = True
    queue: deque[tuple[int, int]] = deque([(i0, j0)])
    cluster: list[tuple[int, int]] = [(i0, j0)]

    while queue:
        ci, cj = queue.popleft()
        for ni, nj in (((ci + 1) % L, cj), ((ci - 1) % L, cj), (ci, (cj + 1) % L), (ci, (cj - 1) % L)):
            if not visited[ni, nj] and int(s[ni, nj]) == spin0:
                if rng.random() < p_add:
                    visited[ni, nj] = True
                    queue.append((ni, nj))
                    cluster.append((ni, nj))

    for ci, cj in cluster:
        s[ci, cj] = -spin0
    return s, len(cluster)
