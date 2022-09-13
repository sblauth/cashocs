# Copyright (C) 2020-2022 Sebastian Blauth
#
# This file is part of cashocs.
#
# cashocs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cashocs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cashocs.  If not, see <https://www.gnu.org/licenses/>.

"""Implementation of a shape optimization problem."""

from __future__ import annotations

import json
import os
import subprocess  # nosec B404
import sys
import tempfile
from types import TracebackType
from typing import Any, Dict, List, Optional, Type, TYPE_CHECKING, Union

import dolfin.function.argument
import fenics
import numpy as np
import ufl
import ufl.algorithms

from cashocs import _exceptions
from cashocs import _forms
from cashocs import _loggers
from cashocs import _pde_problems
from cashocs import geometry
from cashocs._optimization import cost_functional
from cashocs._optimization import line_search
from cashocs._optimization import optimization_algorithms
from cashocs._optimization import optimization_problem
from cashocs._optimization import verification
from cashocs._optimization.shape_optimization import shape_variable_abstractions

if TYPE_CHECKING:
    from cashocs import io
    from cashocs import types


class ShapeOptimizationProblem(optimization_problem.OptimizationProblem):
    r"""A shape optimization problem.

    This class is used to define a shape optimization problem, and to solve
    it subsequently. For a detailed documentation, we refer to the
    :ref:`tutorial <tutorial_index>`. For easier input, when consider single (state or
    control) variables, these do not have to be wrapped into a list. Note, that in the
    case of multiple variables these have to be grouped into ordered lists, where
    ``state_forms``, ``bcs_list``, ``states``, ``adjoints`` have to have the same order
    (i.e. ``[y1, y2]`` and ``[p1, p2]``, where ``p1`` is the adjoint of ``y1`` and so
    on).
    """

    temp_dict: Optional[Dict]

    def __new__(
        cls,
        state_forms: Union[List[ufl.Form], ufl.Form],
        bcs_list: Union[
            List[List[fenics.DirichletBC]],
            List[fenics.DirichletBC],
            fenics.DirichletBC,
        ],
        cost_functional_form: Union[
            List[types.CostFunctional], types.CostFunctional, List[ufl.Form], ufl.Form
        ],
        states: Union[List[fenics.Function], fenics.Function],
        adjoints: Union[List[fenics.Function], fenics.Function],
        boundaries: fenics.MeshFunction,
        config: Optional[io.Config] = None,
        shape_scalar_product: Optional[ufl.Form] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        adjoint_ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        scalar_tracking_forms: Optional[Union[List[Dict], Dict]] = None,
        min_max_terms: Optional[Union[List[Dict], Dict]] = None,
        desired_weights: Optional[List[float]] = None,
    ) -> ShapeOptimizationProblem:
        """Initializes self.

        Args:
            state_forms: The weak form of the state equation (user implemented). Can be
                either a single UFL form, or a (ordered) list of UFL forms.
            bcs_list: The list of :py:class:`fenics.DirichletBC` objects describing
                Dirichlet (essential) boundary conditions. If this is ``None``, then no
                Dirichlet boundary conditions are imposed.
            cost_functional_form: UFL form of the cost functional. Can also be a list of
                summands of the cost functional
            states: The state variable(s), can either be a :py:class:`fenics.Function`,
                or a list of these.
            adjoints: The adjoint variable(s), can either be a
                :py:class:`fenics.Function`, or a (ordered) list of these.
            boundaries: A :py:class:`fenics.MeshFunction` that indicates the boundary
                markers.
            config: The config file for the problem, generated via
                :py:func:`cashocs.create_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            shape_scalar_product: The bilinear form for computing the shape gradient
                (or gradient deformation). This has to use
                :py:class:`fenics.TrialFunction` and :py:class:`fenics.TestFunction`
                objects to define the weak form, which have to be in a
                :py:class:`fenics.VectorFunctionSpace` of continuous, linear Lagrange
                finite elements. Moreover, this form is required to be symmetric.
            initial_guess: List of functions that act as initial guess for the state
                variables, should be valid input for :py:func:`fenics.assign`. Defaults
                to ``None``, which means a zero initial guess.
            ksp_options: A list of strings corresponding to command line options for
                PETSc, used to solve the state systems. If this is ``None``, then the
                direct solver mumps is used (default is ``None``).
            adjoint_ksp_options: A list of strings corresponding to command line options
                for PETSc, used to solve the adjoint systems. If this is ``None``, then
                the same options as for the state systems are used (default is
                ``None``).
            scalar_tracking_forms: A list of dictionaries that define scalar tracking
                type cost functionals, where an integral value should be brought to a
                desired value. Each dict needs to have the keys ``'integrand'`` and
                ``'tracking_goal'``. Default is ``None``, i.e., no scalar tracking terms
                are considered.
            min_max_terms: Additional terms for the cost functional, not to be used
                directly.
            desired_weights: A list of values for scaling the cost functional terms. If
                this is supplied, the cost functional has to be given as list of
                summands. The individual terms are then scaled, so that term `i` has the
                magnitude of `desired_weights[i]` for the initial iteration. In case
                that `desired_weights` is `None`, no scaling is performed. Default is
                `None`.

        """
        use_scaling = bool(desired_weights is not None)

        if use_scaling:
            unscaled_problem = super().__new__(cls)
            unscaled_problem.__init__(  # type: ignore
                state_forms,
                bcs_list,
                cost_functional_form,
                states,
                adjoints,
                boundaries,
                config=config,
                shape_scalar_product=shape_scalar_product,
                initial_guess=initial_guess,
                ksp_options=ksp_options,
                adjoint_ksp_options=adjoint_ksp_options,
                scalar_tracking_forms=scalar_tracking_forms,
                min_max_terms=min_max_terms,
                desired_weights=desired_weights,
            )
            unscaled_problem._scale_cost_functional()  # overwrites the list

            problem = super().__new__(cls)
            if not unscaled_problem.has_cashocs_remesh_flag:
                problem.initial_function_values = (
                    unscaled_problem.initial_function_values
                )

            if (
                not unscaled_problem.has_cashocs_remesh_flag
                and unscaled_problem.do_remesh
                and fenics.MPI.rank(fenics.MPI.comm_world) == 0
            ):
                subprocess.run(  # nosec B603, B607
                    ["rm", "-r", unscaled_problem.temp_dir], check=True
                )
                subprocess.run(  # nosec B603, B607
                    ["rm", "-r", unscaled_problem.mesh_handler.remesh_directory],
                    check=True,
                )
            fenics.MPI.barrier(fenics.MPI.comm_world)

            return problem

        else:
            return super().__new__(cls)

    def __init__(
        self,
        state_forms: Union[List[ufl.Form], ufl.Form],
        bcs_list: Union[
            List[List[fenics.DirichletBC]],
            List[fenics.DirichletBC],
            fenics.DirichletBC,
        ],
        cost_functional_form: Union[
            List[types.CostFunctional], types.CostFunctional, List[ufl.Form], ufl.Form
        ],
        states: Union[List[fenics.Function], fenics.Function],
        adjoints: Union[List[fenics.Function], fenics.Function],
        boundaries: fenics.MeshFunction,
        config: Optional[io.Config] = None,
        shape_scalar_product: Optional[ufl.Form] = None,
        initial_guess: Optional[List[fenics.Function]] = None,
        ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        adjoint_ksp_options: Optional[
            Union[types.KspOptions, List[List[Union[str, int, float]]]]
        ] = None,
        scalar_tracking_forms: Optional[Union[List[Dict], Dict]] = None,
        min_max_terms: Optional[Union[List[Dict], Dict]] = None,
        desired_weights: Optional[List[float]] = None,
    ) -> None:
        """Initializes self.

        Args:
            state_forms: The weak form of the state equation (user implemented). Can be
                either a single UFL form, or a (ordered) list of UFL forms.
            bcs_list: The list of :py:class:`fenics.DirichletBC` objects describing
                Dirichlet (essential) boundary conditions. If this is ``None``, then no
                Dirichlet boundary conditions are imposed.
            cost_functional_form: UFL form of the cost functional. Can also be a list of
                summands of the cost functional
            states: The state variable(s), can either be a :py:class:`fenics.Function`,
                or a list of these.
            adjoints: The adjoint variable(s), can either be a
                :py:class:`fenics.Function`, or a (ordered) list of these.
            boundaries: A :py:class:`fenics.MeshFunction` that indicates the boundary
                markers.
            config: The config file for the problem, generated via
                :py:func:`cashocs.create_config`. Alternatively, this can also be
                ``None``, in which case the default configurations are used, except for
                the optimization algorithm. This has then to be specified in the
                :py:meth:`solve <cashocs.OptimalControlProblem.solve>` method. The
                default is ``None``.
            shape_scalar_product: The bilinear form for computing the shape gradient
                (or gradient deformation). This has to use
                :py:class:`fenics.TrialFunction` and :py:class:`fenics.TestFunction`
                objects to define the weak form, which have to be in a
                :py:class:`fenics.VectorFunctionSpace` of continuous, linear Lagrange
                finite elements. Moreover, this form is required to be symmetric.
            initial_guess: List of functions that act as initial guess for the state
                variables, should be valid input for :py:func:`fenics.assign`. Defaults
                to ``None``, which means a zero initial guess.
            ksp_options: A list of strings corresponding to command line options for
                PETSc, used to solve the state systems. If this is ``None``, then the
                direct solver mumps is used (default is ``None``).
            adjoint_ksp_options: A list of strings corresponding to command line options
                for PETSc, used to solve the adjoint systems. If this is ``None``, then
                the same options as for the state systems are used (default is
                ``None``).
            scalar_tracking_forms: A list of dictionaries that define scalar tracking
                type cost functionals, where an integral value should be brought to a
                desired value. Each dict needs to have the keys ``'integrand'`` and
                ``'tracking_goal'``. Default is ``None``, i.e., no scalar tracking terms
                are considered.
            min_max_terms: Additional terms for the cost functional, not to be used
                directly.
            desired_weights: A list of values for scaling the cost functional terms. If
                this is supplied, the cost functional has to be given as list of
                summands. The individual terms are then scaled, so that term `i` has the
                magnitude of `desired_weights[i]` for the initial iteration. In case
                that `desired_weights` is `None`, no scaling is performed. Default is
                `None`.

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

        # Initialize the remeshing behavior, and a temp file
        self.do_remesh = self.config.getboolean("Mesh", "remesh")
        self.temp_dict = None

        self._remesh_init()

        self.boundaries = boundaries

        self.shape_scalar_product = shape_scalar_product
        if shape_scalar_product is None:
            self.deformation_space: Optional[fenics.FunctionSpace] = None
        else:
            self.deformation_space = shape_scalar_product.arguments()[
                0
            ].ufl_function_space()

        if self.shape_scalar_product is not None:
            self.uses_custom_scalar_product = True

        if self.uses_custom_scalar_product and self.config.getboolean(
            "ShapeGradient", "use_p_laplacian"
        ):
            _loggers.warning(
                (
                    "You have supplied a custom scalar product and set the parameter "
                    "``use_p_laplacian`` in the config file."
                    "cashocs will use the supplied scalar product and not the "
                    "p-Laplacian to compute the shape gradient."
                )
            )

        self.is_shape_problem = True
        self.form_handler: _forms.ShapeFormHandler = _forms.ShapeFormHandler(self)

        if (
            self.do_remesh
            and not self.has_cashocs_remesh_flag
            and self.temp_dict is not None
        ):
            self.temp_dict["Regularization"] = {
                "mu_volume": self.form_handler.shape_regularization.mu_volume,
                "mu_surface": self.form_handler.shape_regularization.mu_surface,
                "mu_curvature": self.form_handler.shape_regularization.mu_curvature,
                "mu_barycenter": self.form_handler.shape_regularization.mu_barycenter,
            }

        self.mesh_handler = geometry._MeshHandler(self)

        self.state_spaces = self.form_handler.state_spaces
        self.adjoint_spaces = self.form_handler.adjoint_spaces

        self.state_problem = _pde_problems.StateProblem(
            self.form_handler, self.initial_guess, self.temp_dict
        )
        self.adjoint_problem = _pde_problems.AdjointProblem(
            self.form_handler, self.state_problem, self.temp_dict
        )
        self.gradient_problem = _pde_problems.ShapeGradientProblem(
            self.form_handler, self.state_problem, self.adjoint_problem
        )

        self.reduced_cost_functional = cost_functional.ReducedCostFunctional(
            self.form_handler, self.state_problem
        )

        self.gradient = self.gradient_problem.gradient
        self.objective_value = 1.0

    def _check_remesh_input(self) -> None:
        """Checks if the inputs are valid for remeshing."""
        if not os.path.isfile(os.path.realpath(sys.argv[0])):
            raise _exceptions.CashocsException(
                "Not a valid configuration. "
                "The script has to be the first command line argument."
            )

        try:
            # pylint: disable=protected-access
            if not self.states[0].function_space().mesh()._config_flag:
                raise _exceptions.InputError(
                    "cashocs.import_mesh",
                    "arg",
                    "You must specify a config file as input for remeshing.",
                )
        except AttributeError as attribute_error:  # pragma: no cover
            raise _exceptions.InputError(
                "cashocs.import_mesh",
                "arg",
                "You must specify a config file as input for remeshing.",
            ) from attribute_error

    def _remesh_init(self) -> None:
        """Initializes self for remeshing."""
        if self.do_remesh:

            self._check_remesh_input()

            if not self.has_cashocs_remesh_flag:
                self.directory = os.path.dirname(os.path.realpath(sys.argv[0]))
                if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
                    temp_dir: str = tempfile.mkdtemp(
                        prefix="._cashocs_remesh_temp_", dir=self.directory
                    )
                else:
                    temp_dir = ""
                fenics.MPI.barrier(fenics.MPI.comm_world)
                self.temp_dir: str = fenics.MPI.comm_world.bcast(temp_dir, root=0)

                self._change_except_hook()
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
                except AttributeError:  # this happens for the unscaled problem
                    pass

            else:
                self._change_except_hook()
                with open(
                    f"{self.temp_dir}/temp_dict.json", "r", encoding="utf-8"
                ) as file:
                    self.temp_dict = json.load(file)

    def _erase_pde_memory(self) -> None:
        """Resets the memory of the PDE problems so that new solutions are computed.

        This sets the value of has_solution to False for all relevant PDE problems,
        where memory is stored.
        """
        super()._erase_pde_memory()
        self.mesh_handler.bbtree.build(self.mesh_handler.mesh)
        self.form_handler.update_scalar_product()
        self.gradient_problem.has_solution = False

    def solve(
        self,
        algorithm: Optional[str] = None,
        rtol: Optional[float] = None,
        atol: Optional[float] = None,
        max_iter: Optional[int] = None,
    ) -> None:
        r"""Solves the optimization problem by the method specified in the config file.

        Args:
            algorithm: Selects the optimization algorithm. Valid choices are
                ``'gradient_descent'`` or ``'gd'`` for a gradient descent method,
                ``'conjugate_gradient'``, ``'nonlinear_cg'``, ``'ncg'`` or ``'cg'``
                for nonlinear conjugate gradient methods, and ``'lbfgs'`` or ``'bfgs'``
                for limited memory BFGS methods. This overwrites the value specified
                in the config file. If this is ``None``, then the value in the
                config file is used. Default is ``None``.
            rtol: The relative tolerance used for the termination criterion.
                Overwrites the value specified in the config file. If this
                is ``None``, the value from the config file is taken. Default
                is ``None``.
            atol: The absolute tolerance used for the termination criterion.
                Overwrites the value specified in the config file. If this
                is ``None``, the value from the config file is taken. Default
                is ``None``.
            max_iter: The maximum number of iterations the optimization algorithm
                can carry out before it is terminated. Overwrites the value
                specified in the config file. If this is ``None``, the value from
                the config file is taken. Default is ``None``.

        Notes:
            If either ``rtol`` or ``atol`` are specified as arguments to the ``.solve``
            call, the termination criterion changes to:

            - a purely relative one (if only ``rtol`` is specified), i.e.,

            .. math:: || \nabla J(u_k) || \leq \texttt{rtol} || \nabla J(u_0) ||.

            - a purely absolute one (if only ``atol`` is specified), i.e.,

            .. math:: || \nabla J(u_K) || \leq \texttt{atol}.

            - a combined one if both ``rtol`` and ``atol`` are specified, i.e.,

            .. math::

                || \nabla J(u_k) || \leq \texttt{atol} + \texttt{rtol}
                || \nabla J(u_0) ||

        """
        super().solve(algorithm=algorithm, rtol=rtol, atol=atol, max_iter=max_iter)

        self.optimization_variable_abstractions = (
            shape_variable_abstractions.ShapeVariableAbstractions(self)
        )
        self.line_search = line_search.ArmijoLineSearch(self)

        # TODO: Do not pass the line search (unnecessary)
        if self.algorithm.casefold() == "gradient_descent":
            self.solver = optimization_algorithms.GradientDescentMethod(
                self, self.line_search
            )
        elif self.algorithm.casefold() == "lbfgs":
            self.solver = optimization_algorithms.LBFGSMethod(self, self.line_search)
        elif self.algorithm.casefold() == "conjugate_gradient":
            self.solver = optimization_algorithms.NonlinearCGMethod(
                self, self.line_search
            )
        elif self.algorithm.casefold() == "none":
            raise _exceptions.InputError(
                "cashocs.OptimalControlProblem.solve",
                "algorithm",
                "You did not specify a solution algorithm in your config file. "
                "You have to specify one in the solve "
                "method. Needs to be one of 'gradient_descent' ('gd'), "
                "'lbfgs' ('bfgs'), or 'conjugate_gradient' ('cg').",
            )

        self.solver.run()
        self.solver.post_processing()

    def _change_except_hook(self) -> None:
        """Ensures that temporary files are deleted when an exception occurs.

        This modifies the sys.excepthook command so that it also deletes temp files
        (only needed for remeshing)
        """

        def custom_except_hook(
            exctype: Type[BaseException],
            value: BaseException,
            traceback: TracebackType,
        ) -> Any:  # pragma: no cover
            """A customized hook which is injected when an exception occurs.

            Args:
                exctype: The type of the exception.
                value: The value of the exception.
                traceback: The traceback of the exception.

            """
            _loggers.debug(
                "An exception was raised by cashocs, "
                "deleting the created temporary files."
            )
            if (
                not self.config.getboolean("Debug", "remeshing")
                and fenics.MPI.rank(fenics.MPI.comm_world) == 0
            ):
                subprocess.run(  # nosec B603, B607
                    ["rm", "-r", self.temp_dir], check=True
                )
                subprocess.run(  # nosec B603, B607
                    ["rm", "-r", self.mesh_handler.remesh_directory], check=True
                )
            fenics.MPI.barrier(fenics.MPI.comm_world)
            sys.__excepthook__(exctype, value, traceback)

        sys.excepthook = custom_except_hook  # type: ignore

    def compute_shape_gradient(self) -> List[fenics.Function]:
        """Solves the Riesz problem to determine the shape gradient.

        This can be used for debugging, or code validation.
        The necessary solutions of the state and adjoint systems
        are carried out automatically.

        Returns:
            A list containing the shape gradient.

        """
        self.gradient_problem.solve()

        return self.gradient

    def supply_shape_derivative(self, shape_derivative: ufl.Form) -> None:
        """Overrides the shape derivative of the reduced cost functional.

        This allows users to implement their own shape derivative and use cashocs as a
        solver library only.

        Args:
            shape_derivative: The shape_derivative of the reduced cost functional.

        """
        if (
            not shape_derivative.arguments()[0].ufl_function_space()
            == self.form_handler.deformation_space
        ):
            shape_derivative = ufl.replace(
                shape_derivative,
                {shape_derivative.arguments()[0]: self.form_handler.test_vector_field},
            )

        self.form_handler.setup_assembler(
            self.form_handler.riesz_scalar_product,
            shape_derivative,
            self.form_handler.bcs_shape,
        )

        self.has_custom_derivative = True

    def supply_custom_forms(
        self,
        shape_derivative: ufl.Form,
        adjoint_forms: Union[ufl.Form, List[ufl.Form]],
        adjoint_bcs_list: Union[
            fenics.DirichletBC, List[fenics.DirichletBC], List[List[fenics.DirichletBC]]
        ],
    ) -> None:
        """Overrides both adjoint system and shape derivative with user input.

        This allows the user to specify both the shape_derivative of the reduced cost
        functional and the corresponding adjoint system, and allows them to use cashocs
        as a solver.

        Args:
            shape_derivative: The shape derivative of the reduced (!) cost functional.
            adjoint_forms: The UFL forms of the adjoint system(s).
            adjoint_bcs_list: The list of Dirichlet boundary conditions for the adjoint
                system(s).

        """
        self.supply_shape_derivative(shape_derivative)
        self.supply_adjoint_forms(adjoint_forms, adjoint_bcs_list)

    def get_vector_field(self) -> dolfin.function.argument.Argument:
        """Returns the TestFunction for defining shape derivatives.

        Returns:
            The TestFunction object.

        """
        return self.form_handler.test_vector_field

    def gradient_test(
        self,
        h: Optional[fenics.Function] = None,
        rng: Optional[np.random.RandomState] = None,
    ) -> float:
        """Performs a Taylor test to verify that the computed shape gradient is correct.

        Args:
            h: The direction used to compute the directional derivative. If this is
                ``None``, then a random direction is used (default is ``None``).
            rng: A numpy random state for calculating a random direction

        Returns:
            The convergence order from the Taylor test. If this is (approximately) 2
            or larger, everything works as expected.

        """
        return verification.shape_gradient_test(self, h, rng)
