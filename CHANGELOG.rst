Change Log
==========

This is CASHOCS' change log. Note, that only major and minor releases are covered
here as they add new functionality or might change the API. For a documentation
of the maintenance releases, please take a look at
`<https://github.com/sblauth/cashocs/releases>`_.


1.2.0 (December 01, 2020)
-------------------------

- Users can now supply their own bilinear form (or scalar product) for the computation
  of the shape gradient, which is then used instead of the linear elasticity formulation.
  This is documented at `<https://cashocs.readthedocs.io/en/latest/demos/shape_optimization/doc_custom_scalar_product.html>`_.

- Added a curvature regularization term for shape optimization, which can be enabled
  via the config files, similarly to already implemented regularizations. This is
  documented at `<https://cashocs.readthedocs.io/en/latest/demos/shape_optimization/doc_regularization.html>`_.

- cashocs can now scale individual terms of the cost functional if this is desired.
  This allows for a more granular handling of problems with cost functionals
  consisting of multiple terms. This also extends to the regularizations for shape optimization,
  see `<https://cashocs.readthedocs.io/en/latest/demos/shape_optimization/doc_regularization.html>`_.
  This feature is documented at `<https://cashocs.readthedocs.io/en/latest/demos/shape_optimization/doc_scaling.html>`_.

- cashocs now uses the logging module to issue messages for the user. The level of
  verbosity can be controlled via :py:func:`cashocs.set_log_level`.

- New configuration file parameters:

  - Section Regularization:

    - ``factor_curvature`` can be used to specify the weight for the curvature regularization term.

    - ``use_relative_weights`` is a boolean which specifies, whether the weights
      should be used as scaling factor in front of the regularization terms (if this is ``False``),
      or whether they should be used to scale the regularization terms so that they
      have the prescribed value on the initial iteration (if this is ``True``).


1.1.0 (November 13, 2020)
-------------------------

- Added the functionality for cashocs to be used as a solver only, where users can specify
  their custom adjoint equations and (shape) derivatives for the optimization
  problems. This is documented at `<https://cashocs.readthedocs.io/en/latest/demos/cashocs_as_solver/solver_index.html>`_.

- Using ``cashocs.create_config`` is deprecated and replaced by ``cashocs.load_config``,
  but the former will still be supported.

- Configuration files are now not strictly necessary, but still very strongly recommended.

- New configuration file parameters:

  - Section Output:

    - ``result_dir`` can be used to specify where CASHOCS' output files should be placed.


1.0.0 (September 18, 2020)
--------------------------

- Initial release of CASHOCS.
