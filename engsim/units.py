"""Reduced Lennard-Jones units and SI conversion helpers (Argon, Methane, Krypton)."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

KB_SI = 1.380649e-23  # J / K
NA_SI = 6.02214076e23  # mol^-1
AMU_KG = 1.66053906660e-27  # kg


@dataclass(frozen=True)
class LJUnitSystem:
    """Mapping between Lennard-Jones reduced units (sigma=eps=m=kB=1) and SI units."""

    name: str
    sigma_m: float
    eps_over_kB_K: float
    mass_amu: float

    @property
    def eps_J(self) -> float:
        return self.eps_over_kB_K * KB_SI

    @property
    def mass_kg(self) -> float:
        return self.mass_amu * AMU_KG

    @property
    def tau_s(self) -> float:
        """Reduced time unit tau = sigma * sqrt(m / eps) [s]."""
        return float(self.sigma_m * np.sqrt(self.mass_kg / self.eps_J))

    @property
    def pressure_Pa(self) -> float:
        """Reduced pressure unit eps / sigma^3 [Pa]."""
        return float(self.eps_J / (self.sigma_m**3))

    @property
    def viscosity_Pa_s(self) -> float:
        """Reduced viscosity unit sqrt(m * eps) / sigma^2 [Pa s]."""
        return float(np.sqrt(self.mass_kg * self.eps_J) / (self.sigma_m**2))

    @property
    def diffusion_m2_s(self) -> float:
        """Reduced diffusion unit sigma^2 / tau [m^2/s]."""
        return float((self.sigma_m**2) / self.tau_s)

    def T_to_K(self, T_star: float) -> float:
        return float(T_star * self.eps_over_kB_K)

    def P_to_MPa(self, P_star: float) -> float:
        return float(P_star * self.pressure_Pa / 1e6)

    def t_to_ps(self, t_star: float) -> float:
        return float(t_star * self.tau_s * 1e12)


ARGON = LJUnitSystem(name="Argon", sigma_m=3.405e-10, eps_over_kB_K=119.8, mass_amu=39.948)
METHANE = LJUnitSystem(name="Methane", sigma_m=3.730e-10, eps_over_kB_K=148.0, mass_amu=16.043)
KRYPTON = LJUnitSystem(name="Krypton", sigma_m=3.630e-10, eps_over_kB_K=163.1, mass_amu=83.798)
