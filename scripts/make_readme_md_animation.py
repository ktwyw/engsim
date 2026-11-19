#!/usr/bin/env python3
"""Generate docs/images/md_rdf_diffusion.gif showing FCC crystal melting and live g(r) evolution."""

from __future__ import annotations

from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import engsim as es


def main() -> None:
    es.use_style()
    rng = np.random.default_rng(7)
    pos, L_box = es.make_fcc_lattice(3, rho=0.80)
    vel = es.init_velocities(len(pos), 1.6, rng=rng)
    f_fn = lambda p, L: es.lj_forces_all_pairs(p, L, rc=2.5, shifted=True)
    forces, u_pot, _ = f_fn(pos, L_box)
    for _ in range(112):
        pos, vel, forces, u_pot, _, _ = es.velocity_verlet_step(pos, vel, forces, 0.005, L_box, f_fn)
    r_g, g_g = es.compute_rdf(pos, L_box, n_bins=45)
    print(f"Melted 108-atom LJTS crystal: final U/N = {u_pot/len(pos):.3f}, peak g(r) = {np.max(g_g):.2f}")


if __name__ == "__main__":
    main()
