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

"""Module including utility and helper functions.

This module includes utility and helper functions used in CASHOCS. They
might also be interesting for users, so they are part of the public API.
Includes wrappers that allow to shorten the coding for often recurring
actions.
"""

from __future__ import annotations

import argparse
import configparser
import os
from pathlib import Path
from typing import Union, List, Tuple, Optional

import fenics
import numpy as np
import ufl
from petsc4py import PETSc

from ._exceptions import InputError, PETScKSPError
from ._loggers import warning
from .config import Config


class Interpolator:
    """Efficient interpolation between two function spaces.

    This is very useful, if multiple interpolations have to be
    carried out between the same spaces, which is made significantly
    faster by computing the corresponding matrix.
    The function spaces can even be defined on different meshes.

    Notes
    -----

    This class only works properly for continuous Lagrange elements and
    constant, discontinuous Lagrange elements. All other elements raise
    an Exception.

    Examples
    --------
    Here, we consider interpolating from CG1 elements to CG2 elements ::

        from fenics import *
        import cashocs

        mesh, _, _, _, _, _ = cashocs.regular_mesh(25)
        V1 = FunctionSpace(mesh, 'CG', 1)
        V2 = FunctionSpace(mesh, 'CG', 2)

        expr = Expression('sin(2*pi*x[0])', degree=1)
        u = interpolate(expr, V1)

        interp = cashocs.utils.Interpolator(V1, V2)
        interp.interpolate(u)
    """

    def __init__(self, V: fenics.FunctionSpace, W: fenics.FunctionSpace) -> None:
        """
        Parameters
        ----------
        V : fenics.FunctionSpace
            The function space whose objects shall be interpolated.
        W : fenics.FunctionSpace
            The space into which they shall be interpolated.
        """

        if not (
            V.ufl_element().family() == "Lagrange"
            or (
                V.ufl_element().family() == "Discontinuous Lagrange"
                and V.ufl_element().degree() == 0
            )
        ):
            raise InputError(
                "cashocs.utils.Interpolator",
                "V",
                "The interpolator only works with CG n or DG 0 elements",
            )
        if not (
            W.ufl_element().family() == "Lagrange"
            or (
                W.ufl_element().family() == "Discontinuous Lagrange"
                and W.ufl_element().degree() == 0
            )
        ):
            raise InputError(
                "cashocs.utils.Interpolator",
                "W",
                "The interpolator only works with CG n or DG 0 elements",
            )

        self.V = V
        self.W = W
        self.transfer_matrix = fenics.PETScDMCollection.create_transfer_matrix(
            self.V, self.W
        )

    def interpolate(self, u: fenics.Function) -> fenics.Function:
        """Interpolates function to target space.

        The function has to belong to the origin space, i.e., the first argument
        of __init__, and it is interpolated to the destination space, i.e., the
        second argument of __init__. There is no need to call set_allow_extrapolation
        on the function (this is done automatically due to the method).

        Parameters
        ----------
        u : fenics.Function
            The function that shall be interpolated.

        Returns
        -------
        fenics.Function
            The result of the interpolation.
        """

        if not u.function_space() == self.V:
            raise InputError(
                "cashocs.utils.Interpolator.interpolate",
                "u",
                "The input does not belong to the correct function space.",
            )
        v = fenics.Function(self.W)
        v.vector()[:] = (self.transfer_matrix * u.vector())[:]

        return v


def summation(
    x: List[Union[ufl.core.expr.Expr, int, float]]
) -> Union[ufl.core.expr.Expr, int, float]:
    """Sums elements of a list in a UFL friendly fashion.

    This can be used to sum, e.g., UFL forms, or UFL expressions
    that can be used in UFL forms.

    Parameters
    ----------
    x : list[ufl.core.expr.Expr or int or float]
        The list of entries that shall be summed.

    Returns
    -------
    ufl.core.expr.Expr or int or float
        Sum of input (same type as entries of input).

    See Also
    --------
    multiplication : Multiplies the elements of a list.

    Notes
    -----
    For "usual" summation of integers or floats, the built-in sum function
    of python or the numpy variant are recommended. Still, they are
    incompatible with FEniCS objects, so this function should be used for
    the latter.

    Examples
    --------
    The command ::

        a = cashocs.summation([u.dx(i)*v.dx(i)*dx for i in mesh.geometric_dimension()])

    is equivalent to ::

        a = u.dx(0)*v.dx(0)*dx + u.dx(1)*v.dx(1)*dx

    (for a 2D mesh).
    """

    if len(x) == 0:
        y = fenics.Constant(0.0)
        warning("Empty list handed to summation, returning 0.")
    else:
        y = x[0]

        for item in x[1:]:
            y += item

    return y


def multiplication(
    x: List[Union[ufl.core.expr.Expr, int, float]]
) -> Union[ufl.core.expr.Expr, int, float]:
    """Multiplies the elements of a list in a UFL friendly fashion.

    Used to build the product of certain UFL expressions to construct
    a UFL form.

    Parameters
    ----------
    x : list[ufl.core.expr.Expr or int or float]
        The list whose entries shall be multiplied.

    Returns
    -------
    ufl.core.expr.Expr or int or float
        The result of the multiplication.

    See Also
    --------
    summation : Sums elements of a list.

    Examples
    --------
    The command ::

        a = cashocs.multiplication([u.dx(i) for i in range(mesh.geometric_dimension())])

    is equivalent to ::

        a = u.dx(0) * u.dx(1)

    (for a 2D mesh).
    """

    if len(x) == 0:
        y = fenics.Constant(1.0)
        warning("Empty list handed to multiplication, returning 1.")
    else:
        y = x[0]

        for item in x[1:]:
            y *= item

    return y


# deprecated
def create_config(path: str) -> configparser.ConfigParser:
    """Loads a config object from a config file.

    Loads the config from a .ini file via the
    configparser package.

    Parameters
    ----------
    path : str
        The path to the .ini file storing the configuration.

    Returns
    -------
    configparser.ConfigParser
        The output config file, which includes the path
        to the .ini file.


    .. deprecated:: 1.1.0
        This is replaced by :py:func:`load_config <cashocs.load_config>`
        and will be removed in the future.
    """

    warning(
        "DEPRECATION WARNING: cashocs.create_config is replaced by cashocs.load_config and will be removed in the future."
    )
    config = load_config(path)

    return config


def load_config(path: str) -> configparser.ConfigParser:
    """Loads a config object from a config file.

    Loads the config from a .ini file via the
    configparser package.

    Parameters
    ----------
    path : str
        The path to the .ini file storing the configuration.

    Returns
    -------
    configparser.ConfigParser
        The output config file, which includes the path
        to the .ini file.
    """
    if os.path.isfile(path):
        config = Config(path)
    else:
        raise InputError(
            "cashocs.utils.load_config",
            "path",
            "The file you specified does not exist.",
        )

    return config


def create_dirichlet_bcs(
    function_space: fenics.FunctionSpace,
    value: Union[
        fenics.Constant, fenics.Expression, fenics.Function, float, Tuple[float]
    ],
    boundaries: fenics.MeshFunction,
    idcs: Union[List[Union[int, str]], int, str],
    **kwargs,
) -> List[fenics.DirichletBC]:
    """Create several Dirichlet boundary conditions at once.

    Wraps multiple Dirichlet boundary conditions into a list, in case
    they have the same value but are to be defined for multiple boundaries
    with different markers. Particularly useful for defining homogeneous
    boundary conditions.

    Parameters
    ----------
    function_space : fenics.FunctionSpace
        The function space onto which the BCs should be imposed on.
    value : fenics.Constant or fenics.Expression or fenics.Function or float or tuple(float)
        The value of the boundary condition. Has to be compatible with the
        function_space, so that it could also be used as
        ``fenics.DirichletBC(function_space, value, ...)``.
    boundaries : fenics.MeshFunction
        The :py:class:`fenics.MeshFunction` object representing the boundaries.
    idcs : list[int or str] or int or str
        A list of indices / boundary markers that determine the boundaries
        onto which the Dirichlet boundary conditions should be applied to.
        Can also be a single entry for a single boundary. If your mesh file
        is named, then you can also use the names of the boundaries to define the
        boundary conditions.
    **kwargs
        Keyword arguments for fenics.DirichletBC

    Returns
    -------
    list[fenics.DirichletBC]
        A list of DirichletBC objects that represent the boundary conditions.

    Examples
    --------
    Generate homogeneous Dirichlet boundary conditions for all 4 sides of
    the unit square ::

        from fenics import *
        import cashocs

        mesh, _, _, _, _, _ = cashocs.regular_mesh(25)
        V = FunctionSpace(mesh, 'CG', 1)
        bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1,2,3,4])
    """

    mesh = function_space.mesh()

    if not isinstance(idcs, list):
        idcs = [idcs]

    bcs_list = []
    for entry in idcs:
        if isinstance(entry, int):
            bcs_list.append(
                fenics.DirichletBC(function_space, value, boundaries, entry, **kwargs)
            )
        elif isinstance(entry, str):
            try:
                physical_groups = mesh._physical_groups
                if entry in physical_groups["ds"].keys():
                    bcs_list.append(
                        fenics.DirichletBC(
                            function_space,
                            value,
                            boundaries,
                            physical_groups["ds"][entry],
                            **kwargs,
                        )
                    )
                else:
                    raise InputError(
                        "cashocs.create_dirichlet_bcs",
                        "idcs",
                        "The string you have supplied is not associated with a boundary.",
                    )
            except AttributeError:
                raise InputError(
                    "cashocs.create_dirichlet_bcs",
                    "mesh",
                    "The mesh you are using does not support string type boundary conditions. These have to be set in the .msh file.",
                )

    return bcs_list


# deprecated
def create_bcs_list(
    function_space: fenics.FunctionSpace,
    value: Union[
        fenics.Constant, fenics.Expression, fenics.Function, float, Tuple[float]
    ],
    boundaries: fenics.MeshFunction,
    idcs: Union[List[Union[int, str]], int, str],
    **kwargs,
) -> List[fenics.DirichletBC]:
    """Create several Dirichlet boundary conditions at once.

    Wraps multiple Dirichlet boundary conditions into a list, in case
    they have the same value but are to be defined for multiple boundaries
    with different markers. Particularly useful for defining homogeneous
    boundary conditions.

    Parameters
    ----------
    function_space : fenics.FunctionSpace
        The function space onto which the BCs should be imposed on.
    value : fenics.Constant or fenics.Expression or fenics.Function or float or tuple(float)
        The value of the boundary condition. Has to be compatible with the
        function_space, so that it could also be used as
        ``fenics.DirichletBC(function_space, value, ...)``.
    boundaries : fenics.MeshFunction
        The :py:class:`fenics.MeshFunction` object representing the boundaries.
    idcs : list[int] or int
        A list of indices / boundary markers that determine the boundaries
        onto which the Dirichlet boundary conditions should be applied to.
        Can also be a single integer for a single boundary.
    **kwargs
        Keyword arguments for fenics.DirichletBC

    Returns
    -------
    list[fenics.DirichletBC]
            A list of DirichletBC objects that represent the boundary conditions.

    Examples
    --------
    Generate homogeneous Dirichlet boundary conditions for all 4 sides of the
    unit square ::

        from fenics import *
        import cashocs

        mesh, _, _, _, _, _ = cashocs.regular_mesh(25)
        V = FunctionSpace(mesh, 'CG', 1)
        bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1,2,3,4])

    .. deprecated:: 1.5.0
        This is replaced by cashocs.create_dirichlet_bcs and will be removed in the future.
    """

    warning(
        "DEPRECATION WARNING: cashocs.create_bcs_list is replaced by cashocs.create_dirichlet_bcs and will be removed in the future."
    )

    bcs_list = []
    if isinstance(idcs, list):
        for i in idcs:
            bcs_list.append(
                fenics.DirichletBC(function_space, value, boundaries, i, **kwargs)
            )

    elif isinstance(idcs, int):
        bcs_list.append(
            fenics.DirichletBC(function_space, value, boundaries, idcs, **kwargs)
        )

    return bcs_list


def _assemble_petsc_system(
    A_form: ufl.Form,
    b_form: ufl.Form,
    bcs: Optional[Union[fenics.DirichletBC, List[fenics.DirichletBC]]] = None,
    A_tensor: Optional[fenics.PETScMatrix] = None,
    b_tensor: Optional[fenics.PETScVector] = None,
) -> Tuple[PETSc.Mat, PETSc.Vec]:
    """Assembles a system symmetrically and converts objects to PETSc format.

    Parameters
    ----------
    A_form : ufl.form.Form
        The UFL form for the left-hand side of the linear equation.
    b_form : ufl.form.Form
        The UFL form for the right-hand side of the linear equation.
    bcs : None or fenics.DirichletBC or list[fenics.DirichletBC], optional
        A list of Dirichlet boundary conditions.
    A_tensor : fenics.PETScMatrix or None, optional
        A matrix into which the result is assembled. Default is ``None``
    b_tensor : fenics.PETScVector or None, optional
        A vector into which the result is assembled. Default is ``None``

    Returns
    -------
    petsc4py.PETSc.Mat
        The petsc matrix for the left-hand side of the linear equation.
    petsc4py.PETSc.Vec
        The petsc vector for the right-hand side of the linear equation.

    Notes
    -----
    This function always uses the ident_zeros method of the matrix in order to add a one to the diagonal
    in case the corresponding row only consists of zeros. This allows for well-posed problems on the
    boundary etc.
    """

    if A_tensor is None:
        A_tensor = fenics.PETScMatrix()
    if b_tensor is None:
        b_tensor = fenics.PETScVector()
    fenics.assemble_system(
        A_form, b_form, bcs, keep_diagonal=True, A_tensor=A_tensor, b_tensor=b_tensor
    )
    A_tensor.ident_zeros()

    A = A_tensor.mat()
    b = b_tensor.vec()

    return A, b


def _setup_petsc_options(
    ksps: List[PETSc.KSP], ksp_options: List[List[List[str]]]
) -> None:
    """Sets up an (iterative) linear solver.

    This is used to pass user defined command line type options for PETSc
    to the PETSc KSP objects. Here, options[i] is applied to ksps[i]

    Parameters
    ----------
    ksps : list[petsc4py.PETSc.KSP]
        A list of PETSc KSP objects (linear solvers) to which the (command line)
        options are applied to.
    ksp_options : list[list[list[str]]]
        A list of command line options that specify the iterative solver
        from PETSc.

    Returns
    -------
    None
    """

    if not len(ksps) == len(ksp_options):
        raise InputError(
            "cashocs.utils._setup_petsc_options",
            "ksps",
            "Length of ksp_options and ksps does not match.",
        )

    opts = fenics.PETScOptions

    for i in range(len(ksps)):
        opts.clear()

        for option in ksp_options[i]:
            opts.set(*option)

        ksps[i].setFromOptions()


def _solve_linear_problem(
    ksp: Optional[PETSc.KSP] = None,
    A: Optional[PETSc.Mat] = None,
    b: Optional[PETSc.Vec] = None,
    x: Optional[PETSc.Vec] = None,
    ksp_options: Optional[List[List[str]]] = None,
    rtol: Optional[float] = None,
    atol: Optional[float] = None,
) -> PETSc.Vec:
    """Solves a finite dimensional linear problem.

    Parameters
    ----------
    ksp : petsc4py.PETSc.KSP or None, optional
        The PETSc KSP object used to solve the problem. None means that the solver
        mumps is used (default is None).
    A : petsc4py.PETSc.Mat or None, optional
        The PETSc matrix corresponding to the left-hand side of the problem. If
        this is None, then the matrix stored in the ksp object is used. Raises
        an error if no matrix is stored. Default is None.
    b : petsc4py.PETSc.Vec or None, optional
        The PETSc vector corresponding to the right-hand side of the problem.
        If this is None, then a zero right-hand side is assumed, and a zero
        vector is returned. Default is None.
    x : petsc4py.PETSc.Vec or None, optional
        The PETSc vector that stores the solution of the problem. If this is
        None, then a new vector will be created (and returned)
    ksp_options : list or None, optional
        The options for the PETSc ksp object. If this is None (the default) a direct
        method is used
    rtol : float or None, optional
        The relative tolerance used in case an iterative solver is used for solving the
        linear problem. Overrides the specification in the ksp object and ksp_options.
    atol : float or None, optional
        The absolute tolerance used in case an iterative solver is used for solving the
        linear problem. Overrides the specification in the ksp object and ksp_options.

    Returns
    -------
    petsc4py.PETSc.Vec
        The solution vector.
    """

    if ksp is None:
        ksp = PETSc.KSP().create()
        options = [
            ["ksp_type", "preonly"],
            ["pc_type", "lu"],
            ["pc_factor_mat_solver_type", "mumps"],
            ["mat_mumps_icntl_24", 1],
        ]

        _setup_petsc_options([ksp], [options])

    if A is not None:
        ksp.setOperators(A)
    else:
        A = ksp.getOperators()[0]
        if A.size[0] == -1 and A.size[1] == -1:
            raise InputError(
                "cashocs.utils._solve_linear_problem",
                "ksp",
                "The KSP object has to be initialized with some Matrix in case A is None.",
            )

    if b is None:
        return A.getVecs()[0]

    if x is None:
        x, _ = A.getVecs()

    if ksp_options is not None:
        opts = fenics.PETScOptions
        opts.clear()

        for option in ksp_options:
            opts.set(*option)

        ksp.setFromOptions()

    if rtol is not None:
        ksp.rtol = rtol
    if atol is not None:
        ksp.atol = atol
    ksp.solve(b, x)

    if ksp.getConvergedReason() < 0:
        raise PETScKSPError(ksp.getConvergedReason())

    return x


def write_out_mesh(
    mesh: fenics.Mesh, original_msh_file: str, out_msh_file: str
) -> None:
    """Writes out the current mesh as .msh file.

    This method updates the vertex positions in the ``original_gmsh_file``, the
    topology of the mesh and its connections are the same. The original GMSH
    file is kept, and a new one is generated under ``out_mesh_file``.

    Parameters
    ----------
    mesh : fenics.Mesh
        The mesh object in fenics that should be saved as GMSH file.
    original_msh_file : str
        Path to the original GMSH mesh file of the mesh object, has to
        end with .msh.
    out_msh_file : str
        Path (and name) of the output mesh file, has to end with .msh.

    Returns
    -------
    None

    Notes
    -----
    The method only works with GMSH 4.1 file format. Others might also work,
    but this is not tested or ensured in any way.
    """

    if not original_msh_file[-4:] == ".msh":
        raise InputError(
            "cashocs.utils.write_out_mesh",
            "original_msh_file",
            "Format for original_mesh_file is wrong, has to end in .msh",
        )
    if not out_msh_file[-4:] == ".msh":
        raise InputError(
            "cashocs.utils.write_out_mesh",
            "out_msh_file",
            "Format for out_mesh_file is wrong, has to end in .msh",
        )

    dim = mesh.geometric_dimension()

    if not Path(out_msh_file).parent.is_dir():
        Path(out_msh_file).parent.mkdir(parents=True, exist_ok=True)

    with open(original_msh_file, "r") as old_file, open(out_msh_file, "w") as new_file:

        points = mesh.coordinates()

        node_section = False
        info_section = False
        subnode_counter = 0
        subwrite_counter = 0
        idcs = np.zeros(1, dtype=int)

        for line in old_file:
            if line == "$EndNodes\n":
                node_section = False

            if not node_section:
                new_file.write(line)
            else:
                split_line = line.split(" ")
                if info_section:
                    new_file.write(line)
                    info_section = False
                else:
                    if len(split_line) == 4:
                        num_subnodes = int(split_line[-1][:-1])
                        subnode_counter = 0
                        subwrite_counter = 0
                        idcs = np.zeros(num_subnodes, dtype=int)
                        new_file.write(line)
                    elif len(split_line) == 1:
                        idcs[subnode_counter] = int(split_line[0][:-1]) - 1
                        subnode_counter += 1
                        new_file.write(line)
                    elif len(split_line) == 3:
                        if dim == 2:
                            mod_line = f"{points[idcs[subwrite_counter]][0]:.16f} {points[idcs[subwrite_counter]][1]:.16f} 0\n"
                        elif dim == 3:
                            mod_line = f"{points[idcs[subwrite_counter]][0]:.16f} {points[idcs[subwrite_counter]][1]:.16f} {points[idcs[subwrite_counter]][2]:.16f}\n"
                        else:
                            raise InputError(
                                "cashocs.utils.write_out_mesh",
                                "mesh",
                                "Not a valid dimension for the mesh.",
                            )
                        new_file.write(mod_line)
                        subwrite_counter += 1

            if line == "$Nodes\n":
                node_section = True
                info_section = True


def _optimization_algorithm_configuration(
    config: configparser.ConfigParser, algorithm: Optional[str] = None
) -> str:
    """Returns the internal name of the optimization algorithm and updates config.

    Parameters
    ----------
    config : configparser.ConfigParser or None
        The config of the problem.
    algorithm : str or None, optional
        A string representing user input for the optimization algorithm
        if this is set via keywords in the .solve() call. If this is
        ``None``, then the config is used to return a consistent value
        for internal use. (Default is None).

    Returns
    -------
    str
        Internal name of the algorithms.
    """

    internal_algorithm = None

    if algorithm is not None:
        overwrite = True
    else:
        overwrite = False
        algorithm = config.get("OptimizationRoutine", "algorithm", fallback="none")

    if not isinstance(algorithm, str):
        raise InputError(
            "cashocs.utils._optimization_algorithm_configuration",
            "algorithm",
            "Not a valid input type for algorithm. Has to be a string.",
        )

    if algorithm in ["gradient_descent", "gd"]:
        internal_algorithm = "gradient_descent"
    elif algorithm in ["cg", "conjugate_gradient", "ncg", "nonlinear_cg"]:
        internal_algorithm = "conjugate_gradient"
    elif algorithm in ["lbfgs", "bfgs"]:
        internal_algorithm = "lbfgs"
    elif algorithm in ["newton"]:
        internal_algorithm = "newton"
    elif algorithm in ["pdas", "primal_dual_active_set"]:
        internal_algorithm = "pdas"
    elif algorithm == "none":
        internal_algorithm = "none"
    else:
        raise InputError(
            "cashocs.utils._optimization_algorithm_configuration",
            "algorithm",
            "Not a valid choice for the optimization algorithm.\n"
            "	For a gradient descent method, use 'gradient_descent' or 'gd'.\n"
            "	For a nonlinear conjugate gradient method use 'cg', 'conjugate_gradient', 'ncg', or 'nonlinear_cg'.\n"
            "	For a limited memory BFGS method use 'bfgs' or 'lbfgs'.\n"
            "	For a truncated Newton method use 'newton' (optimal control only).\n"
            "	For a primal dual active set method use 'pdas' or 'primal dual active set' (optimal control only).",
        )

    if overwrite:
        config.set("OptimizationRoutine", "algorithm", internal_algorithm)

    return internal_algorithm


def _parse_remesh() -> Tuple[bool, str]:
    """Parses command line arguments for the remeshing flag

    Returns
    -------
    bool, str
        A boolean indicating, whether a remeshing was performed and a string which
        points to the remeshing directory.

    """

    temp_dir = None
    cashocs_remesh_flag = False

    parser = argparse.ArgumentParser(description="test argument parser")
    parser.add_argument(
        "--temp_dir", type=str, help="Location of the temp directory for remeshing"
    )
    parser.add_argument(
        "--cashocs_remesh",
        action="store_true",
        help="Flag which indicates whether remeshing has been performed",
    )
    args = parser.parse_args()

    if args.temp_dir is not None:
        if os.path.isdir(os.path.realpath(args.temp_dir)):
            temp_dir = args.temp_dir
        else:
            raise InputError(
                "Command line options",
                "--temp_dir",
                "The temporary directory for remeshing does not exist.",
            )

    if args.cashocs_remesh:
        cashocs_remesh_flag = True
        if args.temp_dir is None:
            raise InputError(
                "Command line options",
                "--temp_dir",
                "Cannot use command line option --cashocs_remesh without specifying --temp_dir.",
            )
        elif not os.path.isfile(os.path.realpath(f"{temp_dir}/temp_dict.json")):
            raise InputError
    else:
        if args.temp_dir is not None:
            raise InputError(
                "Command line options",
                "--temp_dir",
                "Should not specify --temp_dir when not using --cashocs_remesh.",
            )

    return cashocs_remesh_flag, temp_dir


def enlist(arg: Union[object, List]) -> List:
    """Wraps the input argument into a list, if it isn't a list already.

    Parameters
    ----------
    arg : list or object
        The input argument, which is to wrapped into a list

    Returns
    -------
    list
        The object wrapped into a list

    """

    if isinstance(arg, list):
        return arg
    else:
        return [arg]


def _check_and_enlist_bcs(
    bcs_list: Union[
        fenics.DirichletBC, List[fenics.DirichletBC], List[List[fenics.DirichletBC]]
    ]
) -> List[List[fenics.DirichletBC]]:
    """Enlists DirichletBC objects for cashocs

    Parameters
    ----------
    bcs_list : fenics.DirichletBC or list[fenics.DirichletBC] or list[list[fenics.DirichletBC]]
        The list of DirichletBC objects

    Returns
    -------
    list[list[fenics.DirichletBC]]
        The wrapped list of DirichletBC objects
    """

    if isinstance(bcs_list, fenics.DirichletBC):
        return [[bcs_list]]
    elif isinstance(bcs_list, list) and len(bcs_list) == 0:
        return [bcs_list]
    elif isinstance(bcs_list, list) and isinstance(bcs_list[0], fenics.DirichletBC):
        return [bcs_list]
    elif isinstance(bcs_list, list) and isinstance(bcs_list[0], list):
        return bcs_list
    else:
        raise InputError(
            "cashocs.utils._check_and_enlist_bcs",
            "bcs_list",
            "Type of bcs_list is wrong",
        )


def _check_and_enlist_control_constraints(
    control_constraints: Union[
        List[Union[float, int, fenics.Function]],
        List[List[Union[float, int, fenics.Function]]],
    ]
) -> List[List[Union[float, int, fenics.Function]]]:
    """Wraps control constraints into a list suitable for cashocs.

    Parameters
    ----------
    control_constraints : list[float or int or fenics.Function] or list[list[float or int or fenics.Function]]
        The list of control constraints

    Returns
    -------
    list[list[float or int or fenics.Function]]
        The wrapped list of control constraints
    """

    if isinstance(control_constraints, list) and isinstance(
        control_constraints[0], list
    ):
        return control_constraints
    elif isinstance(control_constraints, list) and not isinstance(
        control_constraints[0], list
    ):
        return [control_constraints]
    else:
        raise InputError(
            "cashocs.utils._check_and_enlist_control_constraints",
            "control_constraints",
            "Type of control_constraints is wrong",
        )


def _check_and_enlist_ksp_options(
    ksp_options: Union[List[List[str]], List[List[List[str]]]]
) -> List[List[List[str]]]:
    """Wraps ksp options into a list suitable for cashocs.

    Parameters
    ----------
    ksp_options : list[list[str]] or list[list[list[str]]]
        The list of ksp options

    Returns
    -------
    list[list[list[str]]]
        The wrapped list of ksp options
    """

    if (
        isinstance(ksp_options, list)
        and isinstance(ksp_options[0], list)
        and isinstance(ksp_options[0][0], str)
    ):
        return [ksp_options[:]]

    elif (
        isinstance(ksp_options, list)
        and isinstance(ksp_options[0], list)
        and isinstance(ksp_options[0][0], list)
    ):
        return ksp_options[:]
    else:
        raise InputError(
            "cashocs.utils._check_and_enlist_ksp_options",
            "ksp_options",
            "Type of ksp_options is wrong.",
        )


def _max(
    a: Union[float, fenics.Function], b: Union[float, fenics.Function]
) -> ufl.core.expr.Expr:
    """Computes the maximum of ``a`` and ``b``

    Parameters
    ----------
    a : float or fenics.Function
        The first parameter
    b : float or fenics.Function
        The second parameter

    Returns
    -------
    ufl.core.expr.Expr
        The maximum of ``a`` and ``b``
    """
    return (a + b + abs(a - b)) / fenics.Constant(2.0)


def _min(
    a: Union[float, fenics.Function], b: Union[float, fenics.Function]
) -> ufl.core.expr.Expr:
    """Computes the minimum of ``a`` and ``b``

    Parameters
    ----------
    a : float or fenics.Function
        The first parameter
    b : float or fenics.Function
        The second parameter

    Returns
    -------
    ufl.core.expr.Expr
        The minimum of ``a`` and ``b``
    """

    return (a + b - abs(a - b)) / fenics.Constant(2.0)


def moreau_yosida_regularization(
    term: ufl.core.expr.Expr,
    gamma: float,
    measure: fenics.Measure,
    lower_threshold: Optional[Union[float, fenics.Function]] = None,
    upper_treshold: Optional[Union[float, fenics.Function]] = None,
    shift_lower: Optional[Union[float, fenics.Function]] = None,
    shift_upper: Optional[Union[float, fenics.Function]] = None,
) -> ufl.Form:
    r"""Implements a Moreau-Yosida regularization of an inequality constraint

    The general form of the inequality is of the form ::

        lower_threshold <= term <= upper_threshold

    which is defined over the region specified in ``measure``.

    In case ``lower_threshold`` or ``upper_threshold`` are ``None``, they are set to
    :math:`-\infty` and :math:`\infty`, respectively.

    Parameters
    ----------
    term : ufl.core.expr.Expr
        The term inside the inequality constraint
    gamma : float
        The weighting factor of the regularization
    measure : fenics.Measure
        The measure over which the inequality constraint is defined
    lower_threshold : float or fenics.Function or None, optional
        The lower threshold for the inequality constraint. In case this is ``None``, the
        lower bound is set to :math:`-\infty`. The default is ``None``
    upper_treshold : float or fenics.Function or None, optional
        The upper threshold for the inequality constraint. In case this is ``None``, the
        upper bound is set to :math:`\infty`. The default is ``None``
    shift_lower : float or fenics.Function or None:
        A shift function for the lower bound of the Moreau-Yosida regularization.
        Should be non-positive. In case this is ``None``, it is set to 0.
        Default is ``None``.
    shift_upper : float or fenics.Function or None:
        A shift function for the upper bound of the Moreau-Yosida regularization.
        Should be non-negative. In case this is ``None``, it is set to 0.
        Default is ``None``.

    Returns
    -------
    ufl.form.Form
        The ufl form of the Moreau-Yosida regularization, to be used in the cost functional.
    """

    if lower_threshold is None and upper_treshold is None:
        raise InputError(
            "cashocs.utils.moreau_yosida_regularization",
            "upper_threshold, lower_threshold",
            "At least one of the threshold parameters has to be defined.",
        )

    if shift_lower is None:
        shift_lower = fenics.Constant(0.0)
    if shift_upper is None:
        shift_upper = fenics.Constant(0.0)

    if lower_threshold is not None:
        reg_lower = (
            fenics.Constant(1 / (2 * gamma))
            * pow(
                _min(
                    shift_lower + fenics.Constant(gamma) * (term - lower_threshold),
                    fenics.Constant(0.0),
                ),
                2,
            )
            * measure
        )
    if upper_treshold is not None:
        reg_upper = (
            fenics.Constant(1 / (2 * gamma))
            * pow(
                _max(
                    shift_upper + fenics.Constant(gamma) * (term - upper_treshold),
                    fenics.Constant(0.0),
                ),
                2,
            )
            * measure
        )

    if upper_treshold is not None and lower_threshold is not None:
        return reg_lower + reg_upper
    elif upper_treshold is None and lower_threshold is not None:
        return reg_lower
    elif upper_treshold is not None and lower_threshold is None:
        return reg_upper
