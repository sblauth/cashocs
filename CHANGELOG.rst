Change Log
==========

This is CASHOCS' change log. Note, that only major and minor releases are covered
here as they add new functionality or might change the API. For a documentation
of the maintenance releases, please take a look at
`<https://github.com/sblauth/cashocs/releases>`_.

1.1.0 (November 13, 2020)
-------------------------

- Added the functionality for cashocs to be used as a solver only, where users can specify
  their custom adjoint equations and (shape) derivatives for the optimization
  problems. This is documented at `https://cashocs.readthedocs.io/en/latest/demos/cashocs_as_solver/solver_index.html`_.

- Using ``cashocs.create_config`` is deprecated and replaced by ``cashocs.load_config``,
  but the former will still be supported.

- Configuration files are now not strictly necessary, but still very strongly recommended.

- New configuration file parameters:

  - Section Output:

    - ``result_dir`` can be used to specify where CASHOCS' output files should be placed.


1.0.0 (September 18, 2020)
--------------------------

- Initial release of CASHOCS.
