"""Quantitative reference checks across all 16 chapters of engsim."""

from __future__ import annotations

import numpy as np

import engsim as es
from engsim import reference as ref

RESULTS: list[tuple[str, str, str, float, float, float, float]] = []


def check(
    section: str,
    name: str,
    reference_label: str,
    value: float,
    expected: float,
    rtol: float = 1e-4,
    atol: float = 1e-8,
) -> None:
    RESULTS.append((section, name, reference_label, float(value), float(expected), float(rtol), float(atol)))


def _build_all_checks() -> None:
    rng = np.random.default_rng(2026)

    # Ch 01: Units, FCC lattice, force gradient & Newton's 3rd law
    check("Ch01", "Argon reduced time tau [ps]", "sigma*sqrt(m/eps) = 2.156 ps", es.ARGON.t_to_ps(1.0), 2.156, rtol=1e-2)
    pos_fcc, L_fcc = es.make_fcc_lattice(3, rho=0.8442)  # N = 108
    f_fcc, u_fcc, w_fcc = es.lj_forces_all_pairs(pos_fcc, L_fcc, rc=2.5, shifted=True)
    check("Ch01", "Newton's 3rd Law net force norm", "Exact zero momentum sum", float(np.linalg.norm(np.sum(f_fcc, axis=0))), 0.0, atol=1e-12)
    # Perturb atom 0 and check finite-difference force -dU/dx
    pos_p = pos_fcc.copy()
    pos_p[0, 0] += 0.07
    f_p, _, _ = es.lj_forces_all_pairs(pos_p, L_fcc, rc=2.5, shifted=True)
    eps = 1e-6
    pos_plus = pos_p.copy(); pos_plus[0, 0] += eps
    pos_minus = pos_p.copy(); pos_minus[0, 0] -= eps
    _, u_plus, _ = es.lj_forces_all_pairs(pos_plus, L_fcc, rc=2.5, shifted=True)
    _, u_minus, _ = es.lj_forces_all_pairs(pos_minus, L_fcc, rc=2.5, shifted=True)
    f_fd = -(u_plus - u_minus) / (2.0 * eps)
    check("Ch01", "LJTS force vs central finite-difference -dU/dx", "Exact analytical gradient", float(f_p[0, 0]), f_fd, rtol=1e-7)

    # Ch 02: Velocity Verlet time-reversibility & O(dt^2) energy conservation
    pos0, L0 = es.make_fcc_lattice(3, rho=0.70)
    vel0 = es.init_velocities(pos0.shape[0], T_target=1.2, rng=rng)
    f_fn = lambda p, L: es.lj_forces_all_pairs(p, L, rc=2.5, shifted=True)
    f0, u0, _ = f_fn(pos0, L0)
    p, v, f = pos0.copy(), vel0.copy(), f0.copy()
    E0 = 0.5 * float(np.sum(v * v)) + u0
    for _ in range(30):
        p, v, f, u, _, _ = es.velocity_verlet_step(p, v, f, 0.005, L0, f_fn)
    E_mid = 0.5 * float(np.sum(v * v)) + u
    check("Ch02", "NVE Velocity Verlet energy conservation (dt=0.005)", "Constant Hamiltonian E0", E_mid, E0, rtol=1e-4)
    # Reverse velocities and step back 30 steps
    v = -v
    for _ in range(30):
        p, v, f, u, _, _ = es.velocity_verlet_step(p, v, f, 0.005, L0, f_fn)
    rev_err = float(np.max(np.abs(es.minimum_image(p - pos0, L0))))
    check("Ch02", "Velocity Verlet exact time-reversibility (30 steps back)", "Exact initial positions", rev_err, 0.0, atol=1e-10)

    # Ch 03: O(N) Linked-cell list & Verlet neighbor list exact force parity with O(N^2)
    pos_big, L_big = es.make_fcc_lattice(4, rho=0.80)  # N = 256, L = 6.84, rc = 2.0 -> 3 cells per dim
    pos_big = es.wrap_positions(pos_big + rng.uniform(-0.08, 0.08, size=pos_big.shape), L_big)
    f_n2, u_n2, w_n2 = es.lj_forces_all_pairs(pos_big, L_big, rc=2.0, shifted=True)
    f_cl, u_cl, w_cl = es.lj_forces_cell_list(pos_big, L_big, rc=2.0, shifted=True)
    vnl = es.VerletNeighborList(rc=2.0, skin=0.35)
    f_vl, u_vl, w_vl = vnl.forces(pos_big, L_big, shifted=True)
    check("Ch03", "Linked-cell force parity vs O(N^2)", "Exact O(N^2) pair loop", float(np.max(np.abs(f_cl - f_n2))), 0.0, atol=1e-12)
    check("Ch03", "Verlet neighbor-list force parity vs O(N^2)", "Exact O(N^2) pair loop", float(np.max(np.abs(f_vl - f_n2))), 0.0, atol=1e-12)
    check("Ch03", "Linked-cell potential energy parity", "Exact O(N^2) potential", u_cl, u_n2, rtol=1e-12)

    # Ch 04: Equipartition & ideal-gas virial pressure limit
    vel_eq = es.init_velocities(108, T_target=1.5, rng=rng, exact_T=True)
    ke_eq, T_eq = es.kinetic_energy_and_temp(vel_eq, remove_com_dof=True)
    check("Ch04", "Equipartition temperature initialization", "Target T* = 1.50", T_eq, 1.5, rtol=1e-12)
    check("Ch04", "Zero center-of-mass drift velocity", "Exact zero momentum", float(np.linalg.norm(np.mean(vel_eq, axis=0))), 0.0, atol=1e-13)

    # Ch 05 & Ch 06: BAOAB Langevin exact configurational variance on 1D harmonic oscillator
    T_ho, gamma_ho, dt_ho = 1.0, 1.0, 0.25
    x2_sum = 0.0
    n_ho = 12000
    f_ho = lambda p, L: (-p, 0.5 * float(np.sum(p * p)), 0.0)
    p_arr, v_arr, f_arr = np.array([[0.0]]), np.array([[0.0]]), np.array([[0.0]])
    for step in range(n_ho):
        p_arr, v_arr, f_arr, _, _, _ = es.baoab_langevin_step(p_arr, v_arr, f_arr, dt_ho, None, T_ho, gamma_ho, f_ho, rng)
        if step >= 2000:
            x2_sum += float(p_arr[0, 0] ** 2)
    check("Ch06", "BAOAB harmonic oscillator <x^2> at large dt = 0.25", "Exact kB*T / k = 1.0", x2_sum / (n_ho - 2000), 1.0, rtol=0.08)

    # Ch 07: Nosé-Hoover Chain-2 ergodicity on 1D harmonic oscillator
    x_nh, v_nh, z1, z2 = 1.0, 0.0, 0.0, 0.0
    x2_nh = 0.0
    n_nh = 15000
    for step in range(n_nh):
        x_nh, v_nh, z1, z2 = es.nose_hoover_chain2_1d_step(x_nh, v_nh, z1, z2, 0.02, 1.0, 1.0, 1.0, lambda q: -q)
        if step >= 3000:
            x2_nh += x_nh * x_nh
    check("Ch07", "2-chain Nosé-Hoover harmonic oscillator <x^2>", "Exact canonical <x^2> = 1.0", x2_nh / (n_nh - 3000), 1.0, rtol=0.12)

    # Ch 09: Single-particle O(N) MC energy delta exact match with full O(N^2) recomputation
    pos_mc, L_mc = es.make_fcc_lattice(3, rho=0.75)
    pos_mc = es.wrap_positions(pos_mc + rng.uniform(-0.1, 0.1, size=pos_mc.shape), L_mc)
    _, u_before, _ = es.lj_forces_all_pairs(pos_mc, L_mc, rc=2.5, shifted=True)
    u_i_old = es.lj_single_particle_energy(pos_mc, 5, pos_mc[5], L_mc, rc=2.5, shifted=True)
    trial_5 = es.wrap_positions(pos_mc[5] + np.array([0.12, -0.09, 0.07]), L_mc)
    u_i_new = es.lj_single_particle_energy(pos_mc, 5, trial_5, L_mc, rc=2.5, shifted=True)
    pos_mc_after = pos_mc.copy(); pos_mc_after[5] = trial_5
    _, u_after, _ = es.lj_forces_all_pairs(pos_mc_after, L_mc, rc=2.5, shifted=True)
    check("Ch09", "O(N) MC delta-U vs full O(N^2) energy difference", "Exact potential energy difference", u_i_new - u_i_old, u_after - u_before, rtol=1e-11)

    # Ch 11: 2D Ising exact Onsager critical temperature constant & ordered energy
    spins_ord = np.ones((16, 16), dtype=int)
    e_ord, m_ord = es.ising_2d_energy_mag(spins_ord)
    check("Ch11", "2D Ising ferromagnetic ground-state energy per spin", "Exact -2 J", e_ord, -2.0, atol=1e-12)
    check("Ch11", "2D Ising Onsager critical temperature T_c", "2 / ln(1 + sqrt(2))", ref.ONSAGER_TC_2D_ISING, 2.269185314, rtol=1e-8)

    # Ch 12: Second virial coefficient Boyle temperature sign check & Widom low-density limit
    B2_low = ref.second_virial_lj_numerical(T=1.0)
    B2_high = ref.second_virial_lj_numerical(T=5.0)
    check("Ch12", "LJ second virial B2(T=1.0) < 0 (attractive)", "Negative below Boyle T~3.42", float(B2_low < 0.0), 1.0, atol=1e-12)
    check("Ch12", "LJ second virial B2(T=5.0) > 0 (repulsive)", "Positive above Boyle T~3.42", float(B2_high > 0.0), 1.0, atol=1e-12)

    # Ch 13: FFT autocorrelation exact match with direct O(M^2) VACF
    sig = rng.normal(0.0, 1.0, size=(64, 8, 3))
    acf_fft = es.autocorrelation_fft(sig)
    acf_0_exact = float(np.mean(np.sum(sig * sig, axis=2)))
    check("Ch13", "FFT VACF at lag t = 0", "Exact mean squared speed <|v|^2>", float(acf_fft[0]), acf_0_exact, rtol=1e-12)

    # Ch 15: Ideal polymer chain <R_e^2>, <R_g^2>, and CABS vs Casassa slit-pore partitioning (large N=200 continuous limit)
    chains = es.sample_freely_jointed_chains(n_chains=2500, n_bonds=200, bond_len=1.0, rng=rng)
    re2_num = float(np.mean(np.sum((chains[:, -1, :] - chains[:, 0, :]) ** 2, axis=1)))
    rg2_num = float(np.mean([es.radius_of_gyration_sq(c) for c in chains]))
    re2_ex, rg2_ex = ref.ideal_chain_rg_re_exact(n_bonds=200, bond_len=1.0)
    check("Ch15", "Ideal freely-jointed chain <R_e^2> (N=200)", "Exact N b^2 = 200.0", re2_num, re2_ex, rtol=0.05)
    check("Ch15", "Ideal freely-jointed chain <R_g^2> (N=200)", "Exact N(N+2)/(6(N+1))", rg2_num, rg2_ex, rtol=0.05)
    H_test = np.array([25.0, 40.0])
    K_cabs = es.cabs_slit_partition_coefficient(chains, H_test)
    K_casassa = ref.casassa_slit_partition_exact(np.sqrt(rg2_ex), H_test)
    check("Ch15", "CABS slit-pore partition coefficient K(H=25)", "Casassa (1967) analytical series", float(K_cabs[0]), float(K_casassa[0]), rtol=0.06)

    # Ch 16: Yeh-Hummer 1/L finite-size diffusion correction formula
    yh = ref.yeh_hummer_correction(T=1.0, eta=2.0, box_L=10.0)
    check("Ch16", "Yeh-Hummer hydrodynamic 1/L correction", "2.837297*T / (6*pi*eta*L)", yh, 2.837297 / (120.0 * np.pi), rtol=1e-10)


_build_all_checks()
