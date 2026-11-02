# cashocs Development Notes

cashocs is a Python library for PDE-constrained shape optimization and optimal control. Its implementation is organized under `cashocs/` into optimization, PDE problems, constraints, geometry, I/O, nonlinear solvers, and utilities. The public API is collected in `cashocs/__init__.py`.

## Setup and Project Guidance

- Read [README.rst](README.rst) for installation and usage, and [CONTRIBUTING.rst](CONTRIBUTING.rst) for contribution expectations.
- cashocs uses legacy FEniCS 2019.1 with UFL, PETSc, and MPI; it is not compatible with dolfinx. Some tests also require Gmsh.
- Preserve public API behavior unless the task explicitly calls for an API change. Follow the existing package boundaries and nearby implementation patterns.
- Configuration is accessed using section/key indexing, for example `config["Section"]["option"]`.

## Validation

While developing, run focused tests as needed for the code being changed. For a quick, broader check, run the non-slow tests from the repository root in serial:

```bash
pytest -m "not slow"
```

This check is faster and catches most issues directly. Run the full test suites only afterwards, for final verification:

```bash
pytest
mpirun -n 2 pytest --randomly-seed=0
```

Run linting, type checking, and formatting through the configured pre-commit hooks:

```bash
pre-commit run --all-files
```

The non-slow check must run in serial; do not run it under MPI. Report focused or non-slow checks as such, not as full-suite validation.

## MPI and PETSc

- Keep collective operations and barriers consistent across all ranks. A rank-local assertion or exception before a collective can deadlock the other ranks.
- Preserve distributed vector ownership/layout when creating PETSc vectors; use the corresponding DOLFIN vector as the layout reference.
- For PETSc callbacks, respect callback-provided storage and existing sign/shift conventions. Add regression coverage for serial and parallel behavior when changing these paths.

## Documentation

- If you implement a new feature, briefly document it in `CHANGELOG.rst` under the current "in development" section like the other features mentioned there.
- Documentation sources are under `docs/source/`; the user-facing guides and examples are maintained there and in `demos/`.
- Documentation builds may not have all runtime dependencies installed. Avoid importing optional runtime modules just to inspect their metadata.
- The documentation can be built with

```bash
cd docs
make clean && make html
```
