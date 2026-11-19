"""Physical conservation laws, statistical identities, and scaling validations for engsim."""

from __future__ import annotations

import unittest
import numpy as np

import engsim as es


class TestPhysicalValidation(unittest.TestCase):
    def test_lj_tail_correction_vanishes_as_rc_infinity(self) -> None:
        u_t, p_t = es.lj_tail_corrections(rho=0.8, rc=100.0)
        self.assertLess(abs(u_t), 1e-5)
        self.assertLess(abs(p_t), 2e-5)

    def test_rdf_ideal_gas_flat_unity(self) -> None:
        rng = np.random.default_rng(123)
        traj = rng.uniform(0.0, 10.0, size=(30, 120, 3))
        r, g = es.compute_rdf(traj, box_L=10.0, n_bins=25)
        # Ideal gas g(r) == 1.0 everywhere
        self.assertTrue(np.isclose(np.mean(g[5:]), 1.0, rtol=0.05))

    def test_block_error_white_noise_invariance(self) -> None:
        rng = np.random.default_rng(42)
        white = rng.normal(0.0, 2.0, size=1024)
        sizes, errs = es.flyvbjerg_petersen_block_errors(white)
        exact_se = 2.0 / np.sqrt(1024)
        self.assertTrue(np.isclose(errs[0], exact_se, rtol=0.10))
        self.assertTrue(np.isclose(np.mean(errs[:5]), exact_se, rtol=0.15))

    def test_green_kubo_exponential_integral(self) -> None:
        dt = 0.01
        t = np.arange(0, 1000) * dt
        acf = es.reference.langevin_vacf_exact(t, T=1.5, gamma=2.0, dim=3)
        D_gk = float(es.green_kubo_integral(acf, dt)[-1] / 3.0)
        D_exact = es.reference.langevin_diffusion_exact(T=1.5, gamma=2.0)
        self.assertTrue(np.isclose(D_gk, D_exact, rtol=1e-3))


if __name__ == "__main__":
    unittest.main()
