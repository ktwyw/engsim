# Changelog

All notable changes to `engsim` are documented in this file.

## [0.1.0] - 2026-11-19

### Added
- Complete 16-chapter open book of executed Jupyter notebooks (`notebooks/part1_nve_md/` through `notebooks/part4_transport_lammps/`).
- Complete 16-chapter executed exercise solution notebooks (`solutions/part1_nve_md/` through `solutions/part4_transport_lammps/`).
- Core `engsim` pure-NumPy Python package (`units`, `lattices`, `potentials`, `neighbors`, `integrators`, `analysis`, `montecarlo`, `polymers`, `plotting`, `reference`).
- Automated verification suite (`tests/reference_checks.py`, `tests/test_reference_checks.py`, `tests/test_book.py`, `tests/test_validation.py`).
- Jupyter Book 2 (`myst.yml`) configuration, documentation (`docs/intro.md`, `docs/notation.md`, `docs/references.md`), gallery extraction script (`scripts/make_readme_gallery.py`), and crystal-melting / $g(r)$ animation script (`scripts/make_readme_md_animation.py`).
