# Molecular simulation from scratch in Python

This book teaches molecular dynamics (MD) and Monte Carlo (MC) simulation by
building every algorithm from scratch in plain Python and NumPy across **16
worked chapters** and **16 executed solution notebooks**, accompanied by a
compact, pure-NumPy package (`engsim`).

Before writing a LAMMPS input script, an engineer should understand why
Velocity Verlet conserves a shadow Hamiltonian while explicit Euler spirals
outward, why a Berendsen thermostat suppresses kinetic-energy fluctuations
while Bussi–Donadio–Parrinello (CSVR) and BAOAB Langevin sample the exact
canonical ensemble, why a single Nosé–Hoover thermostat fails ergodicity on a
harmonic oscillator, and how microscopic autocorrelation functions yield
macroscopic diffusion and shear viscosity.

## Structure of the book

- **Part I — Foundations and microcanonical MD ($NVE$, Chapters 1–4):**
  Lennard-Jones and LJTS pair potentials, reduced units, periodic boundary
  conditions, Velocity Verlet time integration, $O(N)$ linked-cell and Verlet
  neighbor lists, radial distribution functions $g(r)$, virial pressure, and
  Flyvbjerg–Petersen block-error analysis.
- **Part II — Ensembles, thermostats, and barostats ($NVT$ & $NPT$, Chapters 5–8):**
  Berendsen vs Bussi CSVR, Andersen and BAOAB Langevin dynamics, Nosé–Hoover
  and Nosé–Hoover chains, and isotropic $NPT$ barostats.
- **Part III — Monte Carlo simulation and free energy (Chapters 9–12):**
  Metropolis $NVT$-MC, $NPT$-MC and Grand Canonical ($\mu VT$) simulation, the
  2D Ising phase transition (Metropolis vs Wolff cluster flips), and Widom
  test-particle insertion / Thermodynamic Integration.
- **Part IV — Transport, polymers, and bridging to LAMMPS (Chapters 13–16):**
  Einstein MSD vs Green–Kubo VACF self-diffusion, Green–Kubo shear viscosity
  and Müller–Plathe RNEMD, Kremer–Grest FENE-WCA polymer chains and Casassa
  slit-pore partitioning, and a capstone validation benchmark against LAMMPS
  with Yeh–Hummer $1/L$ finite-size scaling.
