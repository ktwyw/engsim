# Notation

All simulations default to dimensionless Lennard-Jones reduced units ($\sigma = 1$, $\varepsilon = 1$, $m = 1$, $k_B = 1$). Physical SI conversions for liquid Argon ($\sigma = 3.405\text{ Å}$, $\varepsilon / k_B = 119.8\text{ K}$, $m = 39.948\text{ u}$, $\tau = \sqrt{m\sigma^2/\varepsilon} = 2.156\text{ ps}$) are provided in `engsim.units`.

| Symbol | Meaning | Reduced Unit |
| :--- | :--- | :--- |
| $B_2(T^*)$ | Second virial coefficient | $\sigma^3$ |
| $C_v(t)$ | Velocity autocorrelation function $\langle \mathbf{v}(0)\cdot\mathbf{v}(t)\rangle$ | $\varepsilon / m$ |
| $D, D_{\text{MSD}}, D_{\text{GK}}$ | Self-diffusion coefficient (Einstein MSD vs Green–Kubo) | $\sigma^2 / \tau$ |
| $\mathbf{f}_i, \mathbf{r}_i, \mathbf{v}_i$ | Force, position, and velocity of particle $i$ | $\varepsilon/\sigma$, $\sigma$, $\sigma/\tau$ |
| $g(r)$ | Radial distribution function | — |
| $H_{\text{NH}}$ | Extended Nosé–Hoover conserved shadow Hamiltonian | $\varepsilon$ |
| $K, U, E$ | Kinetic, potential, and total energy | $\varepsilon$ |
| $K_{\text{slit}}$ | Equilibrium polymer partition coefficient in a slit pore | — |
| $L, V = L^3$ | Cubic periodic simulation box length and volume | $\sigma$, $\sigma^3$ |
| $N, \rho^* = N/V$ | Particle count and reduced number density | —, $\sigma^{-3}$ |
| $P^*, P_{xy}$ | Virial hydrostatic pressure and off-diagonal stress tensor component | $\varepsilon / \sigma^3$ |
| $r_c, r_s, r_v$ | Pair cutoff radius, neighbor-list skin, and Verlet list radius ($r_c + r_s$) | $\sigma$ |
| $R_e, R_g$ | Polymer end-to-end distance and radius of gyration | $\sigma$ |
| $T^*, \beta = 1/T^*$ | Reduced temperature and inverse thermal energy | $\varepsilon / k_B$, $\varepsilon^{-1}$ |
| $u(r), u_{\text{LJTS}}(r)$ | Full and truncated-and-shifted Lennard-Jones 12–6 pair potential | $\varepsilon$ |
| $\gamma, \tau_T$ | Langevin friction coefficient and thermostat relaxation time | $\tau^{-1}$, $\tau$ |
| $\Delta t^*$ | Integration timestep | $\tau$ |
| $\eta$ | Shear viscosity (Green–Kubo / Müller–Plathe RNEMD) | $\sqrt{m\varepsilon}/\sigma^2$ |
| $\mu_{\text{ex}}$ | Excess chemical potential (Widom insertion) | $\varepsilon$ |
| $\zeta$ | Nosé–Hoover extended friction variable | $\tau^{-1}$ |
