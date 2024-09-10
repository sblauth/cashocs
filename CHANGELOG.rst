Release Notes
=============

These are cashocs' release notes. Note, that only major and minor releases are covered
here as they add new functionality or might change the API. For a documentation
of the maintenance releases, please take a look at
`<https://github.com/sblauth/cashocs/releases>`_.

2.3.0 (in development)
----------------------


2.2.0 (August 28, 2024)
-----------------------

* Add a wrapper for PETSc's SNES solver for nonlinear equations. This is used internally in cashocs whenever possible. For the solution of the state system, our own Newton solver is the default for backwards compatibility. Users can use the new SNES backend by specifying :ini:`backend = petsc` in the Section StateSystem of the configuration.

* Allows nesting of PETSc Fieldsplit PCs with the command line option "pc_fieldsplit_%d_fields <a,b,...>, as explained at `<https://petsc.org/main/manualpages/PC/PCFieldSplitSetFields/>`_

* Increase the precision of the Gmsh output from cashocs

* New configuration file parameters:

  * Section StateSystem
  
    * :ini:`backend` specifies which backend is used for solving nonlinear equations. The default is :ini:`backend = cashocs`, which is the "old" behavior, and :ini:`backend = petsc` uses the new PETSc SNES interface.

  * Section ShapeGradient

    * :ini:`test_for_intersections` is used to disable the (post mesh movement) check for intersections of the mesh, which ensures physically reasonable designs. This should not be set to `False`.



2.1.0 (February 6, 2024)
------------------------

* The class :py:class:`cashocs.DeformationHandler` cannot be used anymore. Instead, use the class by calling :py:class:`cashocs.geometry.DeformationHandler`. 

* Add the keyword arguments :python:`pre_callback` and :python:`post_callback` to define callbacks when an optimization problem is instanciated.

* Callbacks can now have one parameter - the parameter supplied when being called during runtime is the optimization problem itself

* Add a new function :py:func:`cashocs.io.extract_mesh_from_xdmf` which extracts (any, not only Gmsh) meshes from some XDMF state file (e.g. written by cashocs) and saves the underlying mesh in Gmsh file format.

* Add a new command line interface :bash:`cashocs-extract_mesh` which can be used to invoke :py:func:`cashocs.io.extract_mesh_from_xdmf`.

* The output routines save the xdmf files not in a folder called `xdmf` anymore, but the folder is called `checkpoints`

* The output parameter :ini:`save_mesh` does now not only save the optimized mesh, but also writes a Gmsh .msh file of the current iterated mesh for each iteration. This is very useful for restarting simulations.

* Added the function :py:func:`cashocs.io.import_function` which can be used to load a XDMF Function with a function space. This is very useful for checkpointing (first, read the saved mesh, define the function space, then call :py:func:`cashocs.io.import_function`.

* :py:func:`cashocs.import_mesh` can now also directly import a Gmsh mesh file. Internally, the mesh is directly converted to xdmf and then read. At the moment, this only supports the conversion mode `physical`.

* Add the kwargs `linear_solver` and (where applicable) `adjoint_linear_solver`. These can be used to define custom python KSP objects via petsc4py, most importantly, custom python-based preconditioners can be used with these. The feature is covered in the undocumented demo "demos/shape_optimization/python_pc".

* Add the kwarg `newton_linearization` to the optimization problem classes. This can be used to specify which (alternative) linearization techniques can be used for solving the nonlinear state systems.

* The function :py:func:`cashocs.newton_solve` does not change the user-specified tolerance (in the ksp_options) anymore, unless the kwarg `inexact=True` is set. This means, that the user can use custom "inexact Newton" schemes (e.g., gain one digit in accuracy) too. The old default was to use the relative tolerance of the nonlinear iteration multiplied with a safety factor (0.1).

* New configuration file parameters:

  * Section LineSearch

    * :ini:`fail_if_not_converged` determines, whether the line search is cancelled once the state system cannot be solved or if a new iterate is tried instead.

  * Section ShapeGradient

    * :ini:`inhomogeneous_exponent` is a float, which specifies an exponent for the inhomogeneous mesh stiffness

  * Section MeshQuality

    * :ini:`remesh_iter` is used to perform a remeshing after a certain amount of iterations.



2.0.0 (May 16, 2023)
--------------------

* cashocs has a new docstyle. It now uses the `pydata-sphinx-theme <https://pydata-sphinx-theme.readthedocs.io/en/latest/>`_.

* Added space mapping methods to cashocs. The space mapping methods can utilize parallelism via MPI.

* Added polynomial based models for computing trial stepsizes in an extended Armijo rule.

* implemented a wrapper for :bash:`cashocs-convert`, so that this can be used from inside python too. Simply call :py:func:`cashocs.convert`.

* :bash:`cashocs-convert` now has a default output argument (which is the same name as the input file). This can be invoked with the :bash:`-o` or :bash:`--outfile flag`.

* :bash:`cashocs-convert` now has an additional quiet flag, which can be invoked with :bash:`-q` or :bash:`--quiet`. Analogously, :py:func:`cashocs.convert` also has a keyword argument :python:`quiet`. These arguments / flags suppress its output.

* cashocs now saves files in XDMF file format for visualization and does not use .pvd files anymore. This greatly reduces the number of files needed and also enables better visualization for remeshing.

* cashocs' print calls now flush the output buffer, which helps when sys.stdout is a file.

* The "hook" methods of cashocs (:python:`pre_hook` and :python:`post_hook`) are renamed to "callback", see, e.g., :py:meth:`inject_pre_callback <cashocs.optimization_problem.OptimizationProblem.inject_pre_callback>`.

* cashocs now uses pathlib over os.path

* cashocs' loggers are now not colored anymore, which makes reading the log easier if one logs to a file

* Added i/o possibilites to read meshes and functions from the data saved in the xdmf files for visualization. This is documented `here <https://cashocs.readthedocs.io/en/stable/user/demos/misc/demo_xdmf_io/>`_.

* Deprecated functions have been removed. In particular, the functions :py:func:`create_bcs_list`, :py:func:`create_config`, :py:func:`damped_newton_solve` are removed. They are replaced by :py:func:`create_dirichlet_bcs <cashocs.create_dirichlet_bcs>`, :py:func:`load_config <cashocs.load_config>`, and :py:func:`newton_solve <cashocs.newton_solve>`.

* The usage of the keyword arguments :python:`scalar_tracking_forms` and :python:`min_max_terms` in :py:class:`ShapeOptimizationProblem <cashocs.ShapeOptimizationProblem>` and :py:class:`OptimalControlProblem <cashocs.OptimalControlProblem>` has been removed. Instead, every cost functional is now passed via the :python:`cost_functional_list` parameter. Scalar tracking forms are now realized via :py:class:`ScalarTrackingFunctional <cashocs.ScalarTrackingFunctional>` and min-max terms via :py:class:`MinMaxFunctional <cashocs.MinMaxFunctional>`, see `<https://cashocs.readthedocs.io/en/stable/user/demos/optimal_control/demo_scalar_control_tracking>`_.

* BFGS methods can now be used in a restarted fashion, if desired.

* The BFGS method can now be used in a damped fashion. This ensures that the inverse Hessian approximation stays positive definite.

* The options for defining parameters which are to be supplied to PETSc KSP objects have changed their datatype: They are now given by (lists of) dictionaries instead of nested lists. For options without a value in the command line (e.g. the option :bash:`-ksp_view`) have a value of :python:`None` in the dictionary (so :python:`'ksp_view': None` can be used inside the dictionary to supply the aforementioned option).

* cashocs now includes a :py:func:`cashocs.linear_solve` that can be used to solve linear problems.

* Optimization problems, constrained problems, space mapping problems, and linear and nonlinear solvers now include the keyword argument :python:`preconditioner_forms`, which is a list of UFL forms used to define the preconditioner matrices for solving the respective PDEs.

* Added different mesh conversion modes for :py:func:`cashocs.convert`, which are :python:`"physical"`, :python:`"geometrical"`, and :python:`"none"`. These are used to either use the physical or geometrical entities of Gmsh for the definition of the boundaries and subdomains (or neither of these).

* Changed configuration file parameters

  * Section OptimizationRoutine

    * :ini:`maximum_iterations` is now called :ini:`max_iter`

  * Section Output

    * :ini:`save_pvd` is now called :ini:`save_state`, functionality is the same

    * :ini:`save_pvd_adjoint` is now called :ini:`save_adjoint`, functionality is the same

    * :ini:`save_pvd_gradient` is now called :ini:`save_gradient`, functionality is the same

  * Section LineSearch

    * The parameters :ini:`initial_stepsize`, :ini:`epsilon_armijo`, :ini:`beta_armijo`, and :ini:`safeguard_stepsize` are moved from the OptimizationRoutine section to the LineSearch section. Their behavior is unaltered.

* New configuration file parameters

  * Section AlgoLBFGS
  
    * :ini:`bfgs_periodic_restart` is an integer parameter. If this is 0 (the default), no restarting is done. If this is >0, then the BFGS method is restarted after as many iterations, as given in the parameter

    * :ini:`damped` is a boolean flag which indicates, whether damping should be used for the BFGS method. The default is :ini:`damped = False`.
  
  * Section LineSearch is a completely new section where the line searches can be configured.
  
    * :ini:`method` is a string parameter, which can take the values :ini:`method = armijo` (which is the default previous line search) and :ini:`method = polynomial` (which are the new models)
    
    * :ini:`polynomial_model` is a string parameter which can be either :ini:`polynomial_model = quadratic` or :ini:`polynomial_model = cubic`. In case this is :ini:`polynomial_model = quadratic`, three values (current function value, directional derivative, and trial function value) are used to generate a quadratic model of the one-dimensional cost functional. If this is :ini:`polynmomial_model = cubic`, a cubic model is generated based on the last two guesses for the stepsize. These models are exactly minimized to get a new trial stepsize and a safeguarding is applied so that the steps remain feasible.
    
    * :ini:`factor_high` is one parameter for the safeguarding, the upper bound for the search interval for the stepsize (this is multiplied with the previous stepsize)
    
    * :ini:`factor_low` is the other parameter for the safeguarding, the lower bound for the search interval for the stepsize (this is multiplied with the previous stepsize)

  * Section Output
    
    * :ini:`precision` is an integer which specifies the precision (number of significant digits) when printing to console or file. Default is, as before, 3 significant digits.

1.8.0 (July 6, 2022)
--------------------

* cashocs now has a better memory efficiency

* The printing and file output of cashocs has been modified to better readable and fit the default console window

* The ksp keyword argument for solver routines in the :python:`_utils` module has been removed. Now, KSP objects can be interfaced only directly via :python:`ksp_options`

* Rename the default branch from "master" to "main"

* Implement the "guard against poor scaling" for the stepsize computation from Kelley, but only for the initial stepsize

* New configuration file parameters

  * Section OptimizationRoutine
  
    * :ini:`safeguard_stepsize` is a boolean parameter which dis-/enables the guard against poor scaling for the initial iteration

    
1.7.0 (April 20, 2022)
----------------------

* MPI Support - cashocs now has full MPI support. All of its features, including remeshing, now work out of the box in parallel. Nearly any script using cashocs can be run in parallel by invoking it via :bash:`mpirun -n p python script.py`, where :bash:`p` is the number of MPI processes. Note, that running in parallel may sometimes cause unexpected behavior as it is not tested as well as the serial usage. If you should encounter any bugs, please report them.


1.6.0 (April 4, 2022)
---------------------

* Added the possibility to define additional constraints for the optimization problems as well as solvers which can be used to solve these new problems. This includes Augmented Lagrangian and Quadratic Penalty methods. This feature is documented at `<https://cashocs.readthedocs.io/en/stable/user/demos/optimal_control/demo_constraints>`_.

* Added the possibility for users to execute their own code before each solution of the state system or after each computation of the gradient with the help of :py:meth:`inject_pre_callback <cashocs.optimization_problem.OptimizationProblem.inject_pre_callback>` and :py:meth:`inject_post_callback <cashocs.optimization_problem.OptimizationProblem.inject_post_callback>`. This is documented at `<https://cashocs.readthedocs.io/en/stable/user/demos/optimal_control/demo_pre_post_callbacks>`_.

* Added the possibility to define boundary conditions for control variables. This is documented at `<https://cashocs.readthedocs.io/en/stable/user/demos/optimal_control/demo_control_boundary_conditions>`_.

* Added new style cost functionals, namely :py:class:`cashocs.IntegralFunctional`, :py:class:`cashocs.ScalarTrackingFunctional` and :py:class:`cashocs.MinMaxFunctional`. These allow for a clearer definition of cost functionals and will replace the keyword arguments :python:`scalar_tracking_forms` and :python:`min_max_terms` in the future. The new style cost functionals allow for greater flexibility and extensibility in the future.

* Added the possibility to choose between a direct and iterative solver for computing (shape) gradients. 

* Reworked the private interface of cashocs for better extensibility. The :python:`utils` submodule is now private. Added a new :py:mod:`cashocs.io` submodule for handling in- and output. 

* Reworked the way configuration files are treated in cashocs. Now, they are validated and an exception is raised if a config is found to be invalid. 

* New configuration file parameters:

  * Section OptimizationRoutine
    
    * :ini:`gradient_method` is either :ini:`gradient_method = direct` or :ini:`gradient_method = iterative` and specifies that the corresponding type of solver is used to compute the gradient.
    
    * :ini:`gradient_tol` specifies the tolerance which is used in case an iterative solver is used to compute the (shape) gradient.

    
1.5.0 (December 22, 2021)
-------------------------

* Major performance increase (particularly for large problems)

* Added support for using the p-Laplacian to compute the shape gradient. 

* cashocs now also imports Gmsh Physical Group information when it is given by strings, which can be used in integration measures (e.g., :python:`dx('part1')` or :python:`ds('inlet')`, or for creating Dirichlet boundary conditions (e.g. :python:`cashocs.create_dirichlet_bcs(V, Constant(0.0), boundaries, 'dirichlet_boundary')`).

* The nonlinear solver (Newton's method) got an additional :python:`inexact` parameter, which allows users to use an inexact Newton's method with iterative solvers. Additionally, users can specify their own Jacobians to be used in Newton's method with the keyword argument :python:`dF`.

* Users can now specify the weight of the scalar tracking terms individually (this is now documented).

* New configuration file parameters:

  * Section ShapeGradient

    * :ini:`use_p_laplacian` is a boolean flag which enables the use of the p-Laplacian for the computation of the shape gradient
    
    * :ini:`p_laplacian_power` is an integer parameter specifying the power p used for the p-Laplacian

    * :ini:`p_laplacian_stabilization` is a float parameter, which acts as stabilization term for the p-Laplacian. This should be positive and small (e.g. 1e-3).

    * :ini:`update_inhomogeneous` is a boolean parameter, which allows to update the cell volume when using :ini:`inhomogeneous = True` in the ShapeGradient section. This makes small elements have a higher stiffness and updates this over the course of the optimization. Default is :ini:`update_inhomogeneous = False`

    
1.4.0 (September 3, 2021)
-------------------------

* Added the possibility to compute the stiffness for the shape gradient based on the distance to the boundary using the eikonal equation

* Cashocs now supports the tracking of scalar quantities, which are given as integrals of the states / controls / geometric properties. Input parameter is :python:`scalar_tracking_forms`, which is a dictionary consisting of :python:`'integrand'`, which is the integrand of the scalar quantity, and :python:`'tracking_goal'`, which is the (scalar) value that shall be achieved. This feature is documented at `<https://cashocs.readthedocs.io/en/stable/user/demos/optimal_control/demo_scalar_control_tracking>`_.

* Fixed a bug concerning cashocs' memory management, which would occur if several OptimizationProblems were created one after the other

* Changed the coding style to "black"

* Switched printing to f-string syntax for better readability

* Config files are now copied when they are passed to OptimizationProblems, so that manipulation of them is only possible before the instance is created

* New configuration file parameters:

  * Section ShapeGradient

    * :ini:`use_distance_mu` is a boolean flag which enables stiffness computation based on distances

    * :ini:`dist_min` and :ini:`dist_max` describe the minimal and maximum distance to the boundary for which a certain stiffness is used (see below)

    * :ini:`mu_min` and :ini:`mu_max` describe the stiffness values: If the boundary distance is smaller than :ini:`dist_min`, then :python:`mu = mu_min` and if the distance is larger than :ini:`dist_max`, we have :python:`mu = mu_max`

    * :ini:`smooth_mu` is a boolean flag, which determines how :python:`mu` is interpolated between :ini:`dist_min` and :ini:`dist_max`: If this is set to `False`, linear interpolation is used, otherwise, a cubic spline is used

    * :ini:`boundaries_dist` is a list of boundary indices to which the distance shall be computed

* Small bugfixes and other improvements:

  * Switched to pseudo random numbers for the tests for the sake of reproduceability

  * fixed some tolerances for the tests

  * replaced :python:`os.system()` calls by :python:`subprocess.run()`


1.3.0 (June 11, 2021)
---------------------

* Improved the remeshing workflow and fixed several smaller bugs concerning it

* New configuration file parameters:

  * Section Output
    
    * :ini:`save_pvd_adjoint` is a boolean flag which allows users to also save adjoint states in paraview format

    * :ini:`save_pvd_gradient` is a boolean flag which allows users to save the (shape) gradient(s) in paraview format

    * :ini:`save_txt` is a boolean flag, which allows users to capture the command line output as .txt file


1.2.0 (December 01, 2020)
-------------------------

* Users can now supply their own bilinear form (or scalar product) for the computation of the shape gradient, which is then used instead of the linear elasticity formulation. This is documented at `<https://cashocs.readthedocs.io/en/stable/user/demos/shape_optimization/demo_custom_scalar_product>`_.

* Added a curvature regularization term for shape optimization, which can be enabled via the config files, similarly to already implemented regularizations. This is documented at `<https://cashocs.readthedocs.io/en/stable/user/demos/shape_optimization/demo_regularization>`_.

* cashocs can now scale individual terms of the cost functional if this is desired. This allows for a more granular handling of problems with cost functionals consisting of multiple terms. This also extends to the regularizations for shape optimization, see `<https://cashocs.readthedocs.io/en/stable/user/demos/shape_optimization/demo_regularization>`_. This feature is documented at `<https://cashocs.readthedocs.io/en/stable/user/demos/shape_optimization/demo_scaling>`_.

* cashocs now uses the logging module to issue messages for the user. The level of verbosity can be controlled via :py:func:`cashocs.set_log_level`.

* New configuration file parameters:

  * Section Regularization:

    * :ini:`factor_curvature` can be used to specify the weight for the curvature regularization term.

    * :ini:`use_relative_weights` is a boolean which specifies, whether the weights should be used as scaling factor in front of the regularization terms (if this is `False`), or whether they should be used to scale the regularization terms so that they have the prescribed value on the initial iteration (if this is `True`).


1.1.0 (November 13, 2020)
-------------------------

* Added the functionality for cashocs to be used as a solver only, where users can specify their custom adjoint equations and (shape) derivatives for the optimization problems. This is documented at `<https://cashocs.readthedocs.io/en/stable/user/demos/cashocs_as_solver/index>`_.

* Using :py:func:`cashocs.create_config` is deprecated and replaced by :py:func:`cashocs.load_config`, but the former will still be supported.

* Configuration files are now not strictly necessary, but still very strongly recommended.

* New configuration file parameters:

  * Section Output:

    * :ini:`result_dir` can be used to specify where cashocs' output files should be placed.


1.0.0 (September 18, 2020)
--------------------------

* Initial release of cashocs.


