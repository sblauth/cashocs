cashocs
=======

.. automodule:: cashocs

   
   .. rubric:: Functions

   .. autosummary::
      :toctree:
   
      create_dirichlet_bcs
      interpolate_levelset_function_to_cells
      compute_mesh_quality
      interval_mesh
      regular_box_mesh
      regular_mesh
      convert
      import_mesh
      load_config
      set_log_level
      linear_solve
      newton_solve
      picard_iteration
      snes_solve
      ts_pseudo_solve
   
   .. rubric:: Classes

   .. autosummary::
      :toctree:
   
      ConstrainedOptimalControlProblem
      ConstrainedShapeOptimizationProblem
      EqualityConstraint
      InequalityConstraint
      DeflatedTopologyOptimizationProblem
      DeflatedOptimalControlProblem
      Functional
      IntegralFunctional
      MinMaxFunctional
      ScalarTrackingFunctional
      OptimalControlProblem
      ShapeOptimizationProblem
      TopologyOptimizationProblem
      Interpolator
      LogLevel
   
.. rubric:: Modules

.. autosummary::
   :toctree:
   :recursive:

   cashocs.geometry
   cashocs.io
   cashocs.log
   cashocs.nonlinear_solvers
   cashocs.space_mapping
   cashocs.verification
