# Copyright (C) 2020-2021 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""Implementation of a shape optimization problem.

"""

import json
import os
import subprocess
import sys
import tempfile

import fenics
import numpy as np
import ufl
from ufl import replace
from ufl.algorithms.estimate_degrees import estimate_total_polynomial_degree

from .methods import NCG, GradientDescent, LBFGS
from .._exceptions import CashocsException, ConfigError, InputError
from .._forms import ShapeFormHandler
from .._interfaces import OptimizationProblem
from .._loggers import debug, warning
from .._pde_problems import AdjointProblem, ShapeGradientProblem, StateProblem
from .._shape_optimization import ReducedShapeCostFunctional
from ..geometry import _MeshHandler
from ..verification import shape_gradient_test


class ShapeOptimizationProblem(OptimizationProblem):
    r"""A shape optimization problem.

    This class is used to define a shape optimization problem, and to solve
    it subsequently. For a detailed documentation, we refer to the :ref:`tutorial <tutorial_index>`.
    For easier input, when consider single (state or control) variables,
    these do not have to be wrapped into a list.
    Note, that in the case of multiple variables these have to be grouped into
    ordered lists, where state_forms, bcs_list, states, adjoints have to have
    the same order (i.e. ``[y1, y2]`` and ``[p1, p2]``, where ``p1`` is the adjoint of ``y1``
    and so on.
    """

    def __new__(cls, *args, **kwargs):
        try:
            desired_weights = kwargs["desired_weights"]
            if desired_weights is not None:
                _use_scaling = True
            else:
                _use_scaling = False
        except KeyError:
            _use_scaling = False

        if _use_scaling:
            unscaled_problem = super().__new__(cls)
            unscaled_problem.__init__(*args, **kwargs)
            unscaled_problem._scale_cost_functional()  # overwrites the cost functional list

            problem = super().__new__(cls)
            if not unscaled_problem.has_cashocs_remesh_flag:
                problem.initial_function_values = (
                    unscaled_problem.initial_function_values
                )
                if unscaled_problem.use_scalar_tracking:
                    problem.initial_scalar_tracking_values = (
                        unscaled_problem.initial_scalar_tracking_values
                    )

            if (
                not unscaled_problem.has_cashocs_remesh_flag
                and unscaled_problem.do_remesh
            ):
                subprocess.run(["rm", "-r", unscaled_problem.temp_dir], check=True)
                subprocess.run(
                    ["rm", "-r", unscaled_problem.mesh_handler.remesh_directory],
                    check=True,
                )

            return problem

        else:
            return super().__new__(cls)

    def __init__(
        self,
        state_forms,
        bcs_list,
        cost_functional_form,
        states,
        adjoints,
        boundaries,
        config=None,
        shape_scalar_product=None,
        initial_guess=None,
        ksp_options=None,
        adjoint_ksp_options=None,
        scalar_tracking_forms=None,
        min_max_terms=None,
        desired_weights=None,
    ):
        """This is used to generate all classes and functionalities. First ensures
        consistent input, afterwards, the solution algorithm is initialized.

        Parameters
        ----------
        state_forms : ufl.form.Form or list[ufl.form.Form]
                The weak form of the state equation (user implemented). Can be either
                a single UFL form, or a (ordered) list of UFL forms.
        bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]] or dolfin.fem.dirichletbc.DirichletBC or None
                The list of DirichletBC objects describing Dirichlet (essential) boundary conditions.
                If this is ``None``, then no Dirichlet boundary conditions are imposed.
        cost_functional_form : ufl.form.Form or list[ufl.form.Form]
                UFL form of the cost functional.
        states : dolfin.function.function.Function or list[dolfin.function.function.Function]
                The state variable(s), can either be a :py:class:`fenics.Function`, or a list of these.
        adjoints : dolfin.function.function.Function or list[dolfin.function.function.Function]
                The adjoint variable(s), can either be a :py:class:`fenics.Function`, or a (ordered) list of these.
        boundaries : dolfin.cpp.mesh.MeshFunctionSizet.MeshFunctionSizet
                :py:class:`fenics.MeshFunction` that indicates the boundary markers.
        config : configparser.ConfigParser or None
                The config file for the problem, generated via :py:func:`cashocs.create_config`.
                Alternatively, this can also be ``None``, in which case the default configurations
                are used, except for the optimization algorithm. This has then to be specified
                in the :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
        shape_scalar_product : ufl.form.Form
                The bilinear form for computing the shape gradient (or gradient deformation).
                This has to use :py:class:`fenics.TrialFunction` and :py:class:`fenics.TestFunction`
                objects to define the weak form, which have to be in a :py:class:`fenics.VectorFunctionSpace`
                of continuous, linear Lagrange finite elements. Moreover, this form is required to be
                symmetric.
        initial_guess : list[dolfin.function.function.Function], optional
                List of functions that act as initial guess for the state variables, should be valid input for :py:func:`fenics.assign`.
                Defaults to ``None``, which means a zero initial guess.
        ksp_options : list[list[str]] or list[list[list[str]]] or None, optional
                A list of strings corresponding to command line options for PETSc,
                used to solve the state systems. If this is ``None``, then the direct solver
                mumps is used (default is ``None``).
        adjoint_ksp_options : list[list[str]] or list[list[list[str]]] or None
                A list of strings corresponding to command line options for PETSc,
                used to solve the adjoint systems. If this is ``None``, then the same options
                as for the state systems are used (default is ``None``).
        """

        super().__init__(
            state_forms,
            bcs_list,
            cost_functional_form,
            states,
            adjoints,
            config,
            initial_guess,
            ksp_options,
            adjoint_ksp_options,
            scalar_tracking_forms,
            min_max_terms,
            desired_weights,
        )

        ### Initialize the remeshing behavior, and a temp file
        self.do_remesh = self.config.getboolean("Mesh", "remesh", fallback=False)
        self.temp_dict = None
        if self.do_remesh:

            if not os.path.isfile(os.path.realpath(sys.argv[0])):
                raise CashocsException(
                    "Not a valid configuration. The script has to be the first command line argument."
                )

            try:
                if __IPYTHON__:
                    warning(
                        "You are running a shape optimization problem with remeshing from ipython. Rather run this using the python command."
                    )
            except NameError:
                pass

            try:
                if (
                    not self.states[0].function_space().mesh()._cashocs_generator
                    == "config"
                ):
                    raise InputError(
                        "cashocs.import_mesh",
                        "arg",
                        "You must specify a config file as input for remeshing.",
                    )
            except AttributeError:
                raise InputError(
                    "cashocs.import_mesh",
                    "arg",
                    "You must specify a config file as input for remeshing.",
                )

            if not self.has_cashocs_remesh_flag:
                self.directory = os.path.dirname(os.path.realpath(sys.argv[0]))
                self.temp_dir = tempfile.mkdtemp(
                    prefix="._cashocs_remesh_temp_", dir=self.directory
                )
                self.__change_except_hook()
                self.temp_dict = {
                    "temp_dir": self.temp_dir,
                    "gmsh_file": self.config.get("Mesh", "gmsh_file"),
                    "geo_file": self.config.get("Mesh", "geo_file"),
                    "OptimizationRoutine": {
                        "iteration_counter": 0,
                        "gradient_norm_initial": 0.0,
                    },
                    "output_dict": {},
                }

                try:
                    if self.use_scaling:
                        self.temp_dict[
                            "initial_function_values"
                        ] = self.initial_function_values
                        if self.use_scalar_tracking:
                            self.temp_dict[
                                "initial_scalar_tracking_values"
                            ] = self.initial_scalar_tracking_values
                except AttributeError:
                    pass

            else:
                self.__change_except_hook()
                with open(f"{self.temp_dir}/temp_dict.json", "r") as file:
                    self.temp_dict = json.load(file)

        ### boundaries
        if isinstance(boundaries, fenics.cpp.mesh.MeshFunctionSizet):
            self.boundaries = boundaries
        else:
            raise InputError(
                "cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem",
                "boundaries",
                "Not a valid type for boundaries.",
            )

        # shape_scalar_product
        self.shape_scalar_product = shape_scalar_product
        if shape_scalar_product is None:
            self.deformation_space = None
        else:
            self.deformation_space = self.shape_scalar_product.arguments()[
                0
            ].ufl_function_space()

        if self.shape_scalar_product is not None:
            self.uses_custom_scalar_product = True

        if self.uses_custom_scalar_product and self.config.getboolean(
            "ShapeGradient", "use_p_laplacian", fallback=False
        ):
            warning(
                "You have supplied a custom scalar product and set the parameter ``use_p_laplacian`` in the config file."
                "cashocs will use the supplied scalar product and not the p-Laplacian to compute the shape gradient."
            )

        self.form_handler = ShapeFormHandler(self)

        if self.do_remesh and not self.has_cashocs_remesh_flag:
            self.temp_dict["Regularization"] = {
                "mu_volume": self.form_handler.regularization.mu_volume,
                "mu_surface": self.form_handler.regularization.mu_surface,
                "mu_curvature": self.form_handler.regularization.mu_curvature,
                "mu_barycenter": self.form_handler.regularization.mu_barycenter,
            }

        self.mesh_handler = _MeshHandler(self)

        self.state_spaces = self.form_handler.state_spaces
        self.adjoint_spaces = self.form_handler.adjoint_spaces

        self.state_problem = StateProblem(
            self.form_handler, self.initial_guess, self.temp_dict
        )
        self.adjoint_problem = AdjointProblem(
            self.form_handler, self.state_problem, self.temp_dict
        )
        self.shape_gradient_problem = ShapeGradientProblem(
            self.form_handler, self.state_problem, self.adjoint_problem
        )

        self.reduced_cost_functional = ReducedShapeCostFunctional(
            self.form_handler, self.state_problem
        )

        self.gradient = self.shape_gradient_problem.gradient
        self.objective_value = 1.0

    def _erase_pde_memory(self):
        """Resets the memory of the PDE problems so that new solutions are computed.

        This sets the value of has_solution to False for all relevant PDE problems,
        where memory is stored.

        Returns
        -------
        None
        """

        super()._erase_pde_memory()
        self.mesh_handler.bbtree.build(self.mesh_handler.mesh)
        self.form_handler.update_scalar_product()
        self.shape_gradient_problem.has_solution = False

    def solve(self, algorithm=None, rtol=None, atol=None, max_iter=None):
        r"""Solves the optimization problem by the method specified in the config file.

        Parameters
        ----------
        algorithm : str or None, optional
                Selects the optimization algorithm. Valid choices are
                ``'gradient_descent'`` or ``'gd'`` for a gradient descent method,
                ``'conjugate_gradient'``, ``'nonlinear_cg'``, ``'ncg'`` or ``'cg'``
                for nonlinear conjugate gradient methods, and ``'lbfgs'`` or ``'bfgs'`` for
                limited memory BFGS methods. This overwrites the value specified
                in the config file. If this is ``None``, then the value in the
                config file is used. Default is ``None``.
        rtol : float or None, optional
                The relative tolerance used for the termination criterion.
                Overwrites the value specified in the config file. If this
                is ``None``, the value from the config file is taken. Default
                is ``None``.
        atol : float or None, optional
                The absolute tolerance used for the termination criterion.
                Overwrites the value specified in the config file. If this
                is ``None``, the value from the config file is taken. Default
                is ``None``.
        max_iter : int or None, optional
                The maximum number of iterations the optimization algorithm
                can carry out before it is terminated. Overwrites the value
                specified in the config file. If this is ``None``, the value from
                the config file is taken. Default is ``None``.

        Returns
        -------
        None

        Notes
        -----
        If either ``rtol`` or ``atol`` are specified as arguments to the solve
        call, the termination criterion changes to:

          - a purely relative one (if only ``rtol`` is specified), i.e.,

          .. math:: || \nabla J(u_k) || \leq \texttt{rtol} || \nabla J(u_0) ||.

          - a purely absolute one (if only ``atol`` is specified), i.e.,

          .. math:: || \nabla J(u_K) || \leq \texttt{atol}.

          - a combined one if both ``rtol`` and ``atol`` are specified, i.e.,

          .. math:: || \nabla J(u_k) || \leq \texttt{atol} + \texttt{rtol} || \nabla J(u_0) ||
        """

        super().solve(algorithm=algorithm, rtol=rtol, atol=atol, max_iter=max_iter)

        if self.algorithm == "gradient_descent":
            self.solver = GradientDescent(self)
        elif self.algorithm == "lbfgs":
            self.solver = LBFGS(self)
        elif self.algorithm == "conjugate_gradient":
            self.solver = NCG(self)
        elif self.algorithm == "none":
            raise InputError(
                "cashocs.OptimalControlProblem.solve",
                "algorithm",
                "You did not specify a solution algorithm in your config file. You have to specify one in the solve "
                "method. Needs to be one of 'gradient_descent' ('gd'), 'lbfgs' ('bfgs'), "
                "or 'conjugate_gradient' ('cg').",
            )
        else:
            raise ConfigError(
                "OptimizationRoutine",
                "algorithm",
                "Not a valid input. Needs to be one "
                "of 'gradient_descent' ('gd'), 'lbfgs' ('bfgs'), or 'conjugate_gradient' ('cg').",
            )

        self.solver.run()
        self.solver.post_processing()

    def __change_except_hook(self):
        """Ensures that temp files are deleted when an exception occurs.

        This modifies the sys.excepthook command so that it also deletes temp files
        (only needed for remeshing)

        Returns
        -------
        None
        """

        def custom_except_hook(exctype, value, traceback):
            """A customized hook which is injected when an exception occurs.

            Parameters
            ----------
            exctype
            value
            traceback

            Returns
            -------

            """
            debug(
                "An exception was raised by cashocs, deleting the created temporary files."
            )
            if not self.config.getboolean("Debug", "remeshing", fallback=False):
                subprocess.run(["rm", "-r", self.temp_dir], check=True)
                subprocess.run(
                    ["rm", "-r", self.mesh_handler.remesh_directory], check=True
                )
            sys.__excepthook__(exctype, value, traceback)

        sys.excepthook = custom_except_hook

    def compute_shape_gradient(self):
        """Solves the Riesz problem to determine the shape gradient.

        This can be used for debugging, or code validation.
        The necessary solutions of the state and adjoint systems
        are carried out automatically.

        Returns
        -------
        dolfin.function.function.Function
                The shape gradient.
        """

        self.shape_gradient_problem.solve()

        return self.gradient

    def supply_shape_derivative(self, shape_derivative):
        """Overrides the shape derivative of the reduced cost functional.

        This allows users to implement their own shape derivative and use cashocs as a
        solver library only.

        Parameters
        ----------
        shape_derivative : ufl.form.Form
                The shape_derivative of the reduced (!) cost functional w.r.t. controls.

        Returns
        -------
        None
        """

        try:
            if not (isinstance(shape_derivative, ufl.form.Form)):
                raise InputError(
                    "cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_shape_derivative",
                    "shape_derivative",
                    "shape_derivative have to be a ufl form",
                )
        except:
            raise InputError(
                "cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_shape_derivative",
                "shape_derivative",
                "shape_derivative has to be a ufl form",
            )

        if len(shape_derivative.arguments()) == 2:
            raise InputError(
                "cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_shape_derivative",
                "shape_derivative",
                "Do not use TrialFunction for the shape_derivative.",
            )
        elif len(shape_derivative.arguments()) == 0:
            raise InputError(
                "cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_shape_derivative",
                "shape_derivative",
                "The specified shape_derivative must include a TestFunction object.",
            )

        if (
            not shape_derivative.arguments()[0].ufl_function_space().ufl_element()
            == self.form_handler.deformation_space.ufl_element()
        ):
            raise InputError(
                "cashocs._shape_optimization.shape_optimization_problem.ShapeOptimizationProblem.supply_shape_derivative",
                "shape_derivative",
                "The TestFunction has to be chosen from the same space as the corresponding adjoint.",
            )

        if (
            not shape_derivative.arguments()[0].ufl_function_space()
            == self.form_handler.deformation_space
        ):
            shape_derivative = replace(
                shape_derivative,
                {shape_derivative.arguments()[0]: self.form_handler.test_vector_field},
            )

        if self.form_handler.degree_estimation:
            estimated_degree = np.maximum(
                estimate_total_polynomial_degree(
                    self.form_handler.riesz_scalar_product
                ),
                estimate_total_polynomial_degree(shape_derivative),
            )
            self.form_handler.assembler = fenics.SystemAssembler(
                self.form_handler.riesz_scalar_product,
                shape_derivative,
                self.form_handler.bcs_shape,
                form_compiler_parameters={"quadrature_degree": estimated_degree},
            )
        else:
            try:
                self.form_handler.assembler = fenics.SystemAssembler(
                    self.form_handler.riesz_scalar_product,
                    shape_derivative,
                    self.form_handler.bcs_shape,
                )
            except (AssertionError, ValueError):
                estimated_degree = np.maximum(
                    estimate_total_polynomial_degree(
                        self.form_handler.riesz_scalar_product
                    ),
                    estimate_total_polynomial_degree(shape_derivative),
                )
                self.form_handler.assembler = fenics.SystemAssembler(
                    self.form_handler.riesz_scalar_product,
                    shape_derivative,
                    self.form_handler.bcs_shape,
                    form_compiler_parameters={"quadrature_degree": estimated_degree},
                )

        self.has_custom_derivative = True

    def supply_custom_forms(self, shape_derivative, adjoint_forms, adjoint_bcs_list):
        """Overrides both adjoint system and shape derivative with user input.

        This allows the user to specify both the shape_derivative of the reduced cost functional
        and the corresponding adjoint system, and allows them to use cashocs as a solver.

        See Also
        --------
        supply_shape_derivative
        supply_adjoint_forms

        Parameters
        ----------
        shape_derivative : ufl.form.Form
                The shape derivative of the reduced (!) cost functional.
        adjoint_forms : ufl.form.Form or list[ufl.form.Form]
                The UFL forms of the adjoint system(s).
        adjoint_bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]] or dolfin.fem.dirichletbc.DirichletBC or None
                The list of Dirichlet boundary conditions for the adjoint system(s).

        Returns
        -------
        None
        """

        self.supply_shape_derivative(shape_derivative)
        self.supply_adjoint_forms(adjoint_forms, adjoint_bcs_list)

    def get_vector_field(self):
        """Returns the TestFunction for defining shape derivatives.

        See Also
        --------
        supply_shape_derivative

        Returns
        -------
         : dolfin.function.argument.Argument
                The TestFunction object.
        """

        return self.form_handler.test_vector_field

    def gradient_test(self, h=None, rng=None):
        """Taylor test to verify that the computed shape gradient is correct.

        Parameters
        ----------
        h : dolfin.function.function.Function, optional
            The direction used to compute the directional derivative. If this is
            ``None``, then a random direction is used (default is ``None``).
        rng : numpy.random.RandomState
            A numpy random state for calculating a random direction

        Returns
        -------
         : float
            The convergence order from the Taylor test. If this is (approximately) 2 or larger,
            everything works as expected.
        """
        return shape_gradient_test(self, h, rng)
