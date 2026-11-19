"""MD time integrators (Velocity Verlet, BAOAB Langevin) and thermostats/barostats (Berendsen, Bussi CSVR, Andersen, NH/NHC)."""

from __future__ import annotations

from typing import Callable
import numpy as np

from .potentials import wrap_positions


def kinetic_energy_and_temp(vel: np.ndarray, remove_com_dof: bool = True) -> tuple[float, float]:
    """Compute total kinetic energy K and instantaneous kinetic temperature T_inst."""
    n_atoms, dim = vel.shape
    ke = 0.5 * float(np.sum(vel * vel))
    dof = dim * (n_atoms - 1) if (remove_com_dof and n_atoms > 1) else dim * n_atoms
    T_inst = 2.0 * ke / max(dof, 1)
    return ke, float(T_inst)


def velocity_verlet_step(
    pos: np.ndarray,
    vel: np.ndarray,
    forces: np.ndarray,
    dt: float,
    box_L: float,
    force_fn: Callable[[np.ndarray, float], tuple[np.ndarray, float, float]],
    unwrapped: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray | None]:
    """Single symplectic Velocity Verlet step (m = 1)."""
    vel_half = vel + 0.5 * dt * forces
    dr = dt * vel_half
    pos_new = wrap_positions(pos + dr, box_L)
    unwrapped_new = None if unwrapped is None else (unwrapped + dr)
    forces_new, u_pot, virial = force_fn(pos_new, box_L)
    vel_new = vel_half + 0.5 * dt * forces_new
    return pos_new, vel_new, forces_new, u_pot, virial, unwrapped_new


def berendsen_thermostat(vel: np.ndarray, T_target: float, dt: float, tau_T: float) -> np.ndarray:
    """Berendsen weak-coupling velocity rescaling (deterministic, suppresses canonical variance)."""
    _, T_curr = kinetic_energy_and_temp(vel)
    if T_curr <= 1e-16:
        return vel
    lam = np.sqrt(1.0 + (dt / tau_T) * (T_target / T_curr - 1.0))
    return vel * lam


def bussi_csvr_thermostat(
    vel: np.ndarray,
    T_target: float,
    dt: float,
    tau_T: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Bussi-Donadio-Parrinello (2007) Canonical Sampling through Velocity Rescaling (CSVR)."""
    n_atoms, dim = vel.shape
    dof = dim * (n_atoms - 1) if n_atoms > 1 else dim * n_atoms
    ke = 0.5 * float(np.sum(vel * vel))
    if ke <= 1e-16:
        return vel
    ke_target = 0.5 * dof * T_target
    c = float(np.exp(-dt / tau_T))
    R1 = float(rng.normal(0.0, 1.0))
    # Sum of squares of (dof - 1) independent standard normals = chi^2(dof - 1) = 2 * Gamma((dof - 1)/2, 1)
    if dof > 1:
        sum_r2 = 2.0 * float(rng.gamma(0.5 * (dof - 1), 1.0))
    else:
        sum_r2 = 0.0
    alpha2 = c + (1.0 - c) * (ke_target / (dof * ke)) * (R1 * R1 + sum_r2) + 2.0 * R1 * np.sqrt(c * (1.0 - c) * (ke_target / (dof * ke)))
    alpha = np.sqrt(max(alpha2, 1e-12))
    return vel * alpha


def andersen_thermostat(
    vel: np.ndarray,
    T_target: float,
    dt: float,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Andersen stochastic collision thermostat (collision probability nu * dt per particle)."""
    n_atoms, dim = vel.shape
    collide = rng.random(n_atoms) < (nu * dt)
    if np.any(collide):
        vel = vel.copy()
        vel[collide] = rng.normal(0.0, np.sqrt(T_target), size=(int(np.sum(collide)), dim))
    return vel


def baoab_langevin_step(
    pos: np.ndarray,
    vel: np.ndarray,
    forces: np.ndarray,
    dt: float,
    box_L: float | None,
    T_target: float,
    gamma: float,
    force_fn: Callable[[np.ndarray, float | None], tuple[np.ndarray, float, float]],
    rng: np.random.Generator,
    unwrapped: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray | None]:
    """Symmetric BAOAB Langevin dynamics step (Leimkuhler & Matthews, exact configurational sampling)."""
    # B: half-step kick
    v = vel + 0.5 * dt * forces
    # A: half-step drift
    dr1 = 0.5 * dt * v
    p = pos + dr1
    # O: exact Ornstein-Uhlenbeck velocity step
    c1 = float(np.exp(-gamma * dt))
    c2 = float(np.sqrt(max(1.0 - c1 * c1, 0.0) * T_target))
    v = c1 * v + c2 * rng.normal(0.0, 1.0, size=vel.shape)
    # A: second half-step drift
    dr2 = 0.5 * dt * v
    p_new = (p + dr2) if box_L is None else wrap_positions(p + dr2, box_L)
    unwrapped_new = None if unwrapped is None else (unwrapped + dr1 + dr2)
    # Force evaluation & B: second half-step kick
    forces_new, u_pot, virial = force_fn(p_new, box_L)
    v_new = v + 0.5 * dt * forces_new
    return p_new, v_new, forces_new, u_pot, virial, unwrapped_new


def nose_hoover_1d_step(
    x: float,
    v: float,
    zeta: float,
    dt: float,
    T_target: float,
    Q: float,
    force_1d: Callable[[float], float],
) -> tuple[float, float, float]:
    """Symmetric Martyna-Tuckerman operator-split Nosé-Hoover step for a 1D particle (m=1)."""
    # Half-step thermostat
    zeta = zeta + 0.25 * dt * (v * v - T_target) / Q
    v = v * float(np.exp(-0.5 * dt * zeta))
    zeta = zeta + 0.25 * dt * (v * v - T_target) / Q
    # Velocity Verlet core
    v = v + 0.5 * dt * force_1d(x)
    x = x + dt * v
    v = v + 0.5 * dt * force_1d(x)
    # Second half-step thermostat
    zeta = zeta + 0.25 * dt * (v * v - T_target) / Q
    v = v * float(np.exp(-0.5 * dt * zeta))
    zeta = zeta + 0.25 * dt * (v * v - T_target) / Q
    return float(x), float(v), float(zeta)


def nose_hoover_chain2_1d_step(
    x: float,
    v: float,
    zeta1: float,
    zeta2: float,
    dt: float,
    T_target: float,
    Q1: float,
    Q2: float,
    force_1d: Callable[[float], float],
) -> tuple[float, float, float, float]:
    """Symmetric 2-chain Nosé-Hoover (NHC-2) step for a 1D particle restoring full ergodicity."""
    def nhc_half(v_in: float, z1: float, z2: float, h: float) -> tuple[float, float, float]:
        G2 = (Q1 * z1 * z1 - T_target) / Q2
        z2 += 0.25 * h * G2
        z1 *= float(np.exp(-0.125 * h * z2))
        G1 = (v_in * v_in - T_target) / Q1
        z1 += 0.25 * h * G1
        z1 *= float(np.exp(-0.125 * h * z2))
        v_out = v_in * float(np.exp(-0.5 * h * z1))
        z1 *= float(np.exp(-0.125 * h * z2))
        G1 = (v_out * v_out - T_target) / Q1
        z1 += 0.25 * h * G1
        z1 *= float(np.exp(-0.125 * h * z2))
        G2 = (Q1 * z1 * z1 - T_target) / Q2
        z2 += 0.25 * h * G2
        return v_out, z1, z2

    v, zeta1, zeta2 = nhc_half(v, zeta1, zeta2, dt)
    v = v + 0.5 * dt * force_1d(x)
    x = x + dt * v
    v = v + 0.5 * dt * force_1d(x)
    v, zeta1, zeta2 = nhc_half(v, zeta1, zeta2, dt)
    return float(x), float(v), float(zeta1), float(zeta2)


def berendsen_barostat_scale(
    pos: np.ndarray,
    box_L: float,
    P_curr: float,
    P_target: float,
    dt: float,
    tau_P: float,
    kappa_T: float = 0.05,
) -> tuple[np.ndarray, float]:
    """Isotropic Berendsen pressure coupling scaling box_L and particle coordinates."""
    mu3 = 1.0 - (kappa_T * dt / tau_P) * (P_target - P_curr)
    mu = float(np.clip(mu3, 0.90, 1.10) ** (1.0 / 3.0))
    return pos * mu, float(box_L * mu)
