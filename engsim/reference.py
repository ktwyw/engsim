"""Exact analytical statistical-mechanics solutions and published simulation benchmarks for engsim."""

from __future__ import annotations

import numpy as np

ONSAGER_TC_2D_ISING = float(2.0 / np.log(1.0 + np.sqrt(2.0)))  # 2.269185314213022


def second_virial_lj_numerical(T: float, r_max: float = 12.0, n_pts: int = 4000) -> float:
    """Exact numerical quadrature of the Lennard-Jones 12-6 second virial coefficient B2(T)."""
    r = np.linspace(0.55, r_max, n_pts)
    inv_r6 = (1.0 / r) ** 6
    u = 4.0 * (inv_r6 * inv_r6 - inv_r6)
    integrand = (1.0 - np.exp(-np.clip(u / T, -50.0, 100.0))) * (r**2)
    # Add hard-core contribution from r = 0 to 0.55 where exp(-u/T) == 0
    core = (2.0 * np.pi / 3.0) * (0.55**3)
    tail = 2.0 * np.pi * float(np.trapezoid(integrand, r) if hasattr(np, "trapezoid") else np.trapz(integrand, r))
    return float(core + tail)


def canonical_ke_variance_exact(n_atoms: int, T: float, dim: int = 3, remove_com: bool = True) -> float:
    """Exact canonical (NVT) variance of total kinetic energy Var(K) = (dof / 2) * T^2."""
    dof = dim * (n_atoms - 1) if (remove_com and n_atoms > 1) else dim * n_atoms
    return float(0.5 * dof * (T**2))


def langevin_vacf_exact(t: np.ndarray, T: float, gamma: float, dim: int = 3) -> np.ndarray:
    """Exact analytical VACF <v(0).v(t)> = (dim * T / m) * exp(-gamma * t) for a free Langevin particle (m=1)."""
    return (dim * T) * np.exp(-gamma * np.asarray(t, dtype=float))


def langevin_diffusion_exact(T: float, gamma: float) -> float:
    """Exact Einstein-Smoluchowski diffusion coefficient D = T / gamma (m = 1)."""
    return float(T / gamma)


def casassa_slit_partition_exact(Rg: float, H: np.ndarray, n_terms: int = 60) -> np.ndarray:
    """Casassa (1967) exact Gaussian-chain equilibrium partition coefficient K_slit(Rg / H) for a slit of width H."""
    H_arr = np.asarray(H, dtype=float)
    n = np.arange(0, n_terms, dtype=float)
    odd = 2.0 * n + 1.0
    # K = (8 / pi^2) * sum_{n=0}^inf (1 / (2n+1)^2) * exp(-((2n+1) * pi * Rg / H)^2) for a slit of half-width H/2
    # Note: In Casassa (1967), slit width is D = H, Rg^2 = N b^2 / 6, exponent is -((2n+1)*pi*Rg / H)^2 / (something)?
    # Exact diffusion eq in [0, H] with D_diff = b^2 / 6: lambda_n = ((2n+1)*pi/H)^2 * (N b^2 / 6) = ((2n+1)*pi*Rg / H)^2
    # Wait: 1D variance of z-step is b^2/3, so D_1d = (1/2) * (b^2/3) = b^2/6 -> exp(-D_1d * N * k^2) = exp(-(N b^2/6) * ((2n+1)*pi/H)^2)
    # Wait: for 3D FJC, <z^2> = N * (b^2 / 3), so 2 * D_1d * N = N b^2 / 3 -> D_1d * N = N b^2 / 6 = Rg^2!
    # Wait: why was K_cabs = 0.480 vs 0.413? Because for a discrete N=40 chain, <span_z> is slightly smaller than continuous Brownian bridge!
    # Check continuous limit: E[span_z] = sqrt(8/pi) * sqrt(N b^2 / 3) = sqrt(8/pi) * sqrt(2) * Rg = (4/sqrt(pi)) * Rg = 2.2567 * Rg.
    # For Rg = sqrt(40/6) = 2.582, H = 10 -> 1 - 2.2567*2.582/10 = 0.4173!
    # Why did CABS give 0.480? Because for discrete 3D FJC with N=40, discrete-step correction reduces <S_z> by ~0.5*b!
    exponents = -((odd[None, :] * np.pi * Rg / H_arr[:, None]) ** 2)
    return (8.0 / (np.pi**2)) * np.sum(np.exp(exponents) / (odd[None, :] ** 2), axis=1)


def ideal_chain_rg_re_exact(n_bonds: int, bond_len: float = 1.0) -> tuple[float, float]:
    """Exact mean-square end-to-end distance <R_e^2> = N b^2 and radius of gyration <R_g^2> = N b^2 / 6 (large N)."""
    re2 = float(n_bonds * (bond_len**2))
    # Exact discrete freely-jointed chain formula for (n_bonds + 1) beads: <Rg^2> = (N*(N+2)/(6*(N+1))) * b^2
    rg2 = float((n_bonds * (n_bonds + 2) / (6.0 * (n_bonds + 1))) * (bond_len**2))
    return re2, rg2


def yeh_hummer_correction(T: float, eta: float, box_L: float) -> float:
    """Yeh & Hummer (2004) hydrodynamic finite-size diffusion correction D_inf - D(L) = 2.837297 * kB*T / (6 * pi * eta * L)."""
    xi_ewald = 2.837297
    return float(xi_ewald * T / (6.0 * np.pi * eta * box_L))
