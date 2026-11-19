"""Unit tests for core engsim modules."""

from __future__ import annotations

import unittest
import numpy as np

import engsim as es


class TestEngsimCore(unittest.TestCase):
    def test_fcc_and_sc_lattices(self) -> None:
        p_fcc, L_fcc = es.make_fcc_lattice(3, rho=0.8)
        self.assertEqual(p_fcc.shape, (108, 3))
        self.assertTrue(np.isclose(108.0 / (L_fcc**3), 0.8))
        p_sc, L_sc = es.make_sc_lattice(4, rho=0.5)
        self.assertEqual(p_sc.shape, (64, 3))
        self.assertTrue(np.isclose(64.0 / (L_sc**3), 0.5))

    def test_bussi_csvr_vs_berendsen_variance(self) -> None:
        rng = np.random.default_rng(7)
        T_target = 1.2
        v_csvr = es.init_velocities(64, T_target, rng=rng)
        v_ber = v_csvr.copy()
        ke_csvr, ke_ber = [], []
        for _ in range(1500):
            v_csvr = es.bussi_csvr_thermostat(v_csvr, T_target, dt=0.01, tau_T=0.1, rng=rng)
            v_ber = es.berendsen_thermostat(v_ber, T_target, dt=0.01, tau_T=0.1)
            ke_csvr.append(0.5 * float(np.sum(v_csvr**2)))
            ke_ber.append(0.5 * float(np.sum(v_ber**2)))
        var_exact = es.reference.canonical_ke_variance_exact(64, T_target)
        # Bussi CSVR reproduces canonical kinetic variance; Berendsen collapses it to ~0
        self.assertGreater(np.var(ke_csvr[300:]), 0.5 * var_exact)
        self.assertLess(np.var(ke_ber[300:]), 1e-6 * var_exact)

    def test_fene_bond_force_gradient(self) -> None:
        pos = np.array([[0.0, 0.0, 0.0], [0.95, 0.1, -0.05], [1.90, 0.2, 0.10]])
        f, u = es.fene_bond_forces(pos)
        eps = 1e-6
        p_plus = pos.copy(); p_plus[1, 0] += eps
        p_minus = pos.copy(); p_minus[1, 0] -= eps
        _, u_p = es.fene_bond_forces(p_plus)
        _, u_m = es.fene_bond_forces(p_minus)
        self.assertTrue(np.isclose(f[1, 0], -(u_p - u_m) / (2.0 * eps), rtol=1e-7))
        self.assertTrue(np.allclose(np.sum(f, axis=0), np.zeros(3), atol=1e-13))

    def test_ising_wolff_and_metropolis(self) -> None:
        rng = np.random.default_rng(11)
        spins = np.ones((12, 12), dtype=int)
        s_m = es.ising_2d_metropolis_sweep(spins, beta=0.44, rng=rng)
        s_w, c_size = es.ising_2d_wolff_step(s_m, beta=0.44, rng=rng)
        self.assertGreaterEqual(c_size, 1)
        self.assertEqual(s_w.shape, (12, 12))


if __name__ == "__main__":
    unittest.main()
