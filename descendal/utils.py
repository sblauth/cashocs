"""Module for utility and helper functions.

The functions that may be interesting for the end user, too,
are directly imported in descendal, i.e., create_bcs_list and
create_config. However, summation, multiplication, and measure
manipulation may also be of interest.
"""

import fenics
import configparser
import os
from petsc4py import PETSc



def summation(x):
	"""Sums elements of a list.

	This can be used to sum, e.g., UFL forms, or UFL expressions
	that should be used in UFL forms. This is not possible with
	the built-in sum function.

	See Also
	--------
	multiplication : Multiplies the elements of a list.

	Parameters
	----------
	x : list[ufl.form.Form] or list[int] or list[float]
		The list of entries that shall be summed.

	Returns
	-------
	y : ufl.form.Form or int or float
		Sum of input (same type as entries of input).
	"""
	
	if len(x) == 0:
		y = fenics.Constant(0.0)
		print('Careful, empty list handed to summ')
	else:
		y = x[0]
		
		for item in x[1:]:
			y += item
	
	return y



def multiplication(x):
	"""Multiplies the elements of a list with each other.

	Used to build the product of certain UFL expressions to construct
	a UFL form.

	See Also
	--------
	summation : Sums elements of a list.

	Parameters
	----------
	x : list[ufl.core.expr.Expr] or list[int] or list[float]
		The list whose entries shall be multiplied.

	Returns
	-------
	y : ufl.core.expr.Expr or int or float
		The result of the multiplication.
	"""
	
	if len(x) == 0:
		y = fenics.Constant(1.0)
		print('Careful, empty list handed to multiplication')
	else:
		y = x[0]
		
		for item in x[1:]:
			y *= item
			
	return y



class EmptyMeasure:
	"""Implements an empty measure (e.g. of a null set).

	This is used for automatic measure generation, e.g., if
	the fixed boundary is empty for a shape optimization problem,
	and is used to avoid case distinctions.
	"""

	def __init__(self, measure):
		"""Initializes self.

		Parameters
		----------
		measure : ufl.measure.Measure
			The underlying UFL measure.
		"""

		self.measure = measure



	def __rmul__(self, other):
		"""Generates a UFL form of the empty measure when multiplied from the right.

		Parameters
		----------
		other : ufl.core.expr.Expr
			A UFL expression to be integrated over an empty measure.

		Returns
		-------
		ufl.form.Form
			The resulting UFL form.
		"""

		return fenics.Constant(0)*other*self.measure



def generate_measure(idx, measure):
	"""Generates a measure based on indices.

	Generates a MeasureSum or EmptyMeasure object corresponding to
	measure and the subdomains / boundaries specified in idx. This
	is a convenient shortcut to writing dx(1) + dx(2) + dx(3) + ...
	in case many measures are involved.

	Parameters
	----------
	idx : list[int]
		A list of indices for the boundary / volume markers that
		shall define the new measure.
	measure : ufl.measure.Measure
		The corresponding UFL measure.

	Returns
	-------
	ufl.measure.Measure or descendal.utils.EmptyMeasure
		The corresponding sum of the measures or an empty measure.
	"""

	if len(idx) == 0:
		out_measure = EmptyMeasure(measure)

	else:
		out_measure = measure(idx[0])

		for i in idx[1:]:
			out_measure += measure(i)

	return out_measure



def create_config(path):
	"""Generates a config object from a config file.

	Creates the config from a .ini file via the
	configparser package, and also adds the path of the
	config file to the config. This is helpful for remeshing.

	Parameters
	----------
	path : str
		The path to the config .ini file.

	Returns
	-------
	configparser.ConfigParser
		The output config file, which includes the path
		to the .ini file.
	"""

	config = configparser.ConfigParser()
	config.read(path)
	config.set('Mesh', 'config_path', os.path.abspath(path))

	return config



def create_bcs_list(function_space, value, boundaries, idcs):
	"""Create several Dirichlet boundary conditions at once.

	Wraps multiple Dirichlet boundary conditions into a list, in case
	they have the same value but are to be defined for multiple boundaries
	with different markers. Particularly useful for defining homogeneous
	boundary conditions.

	Parameters
	----------
	function_space : dolfin.function.functionspace.FunctionSpace
		The function space onto which the BCs should be imposed on.
	value : dolfin.function.constant.Constant or dolfin.function.expression.Expression or dolfin.function.function.Function or float or tuple(float)
		The value of the boundary condition. Has to be compatible with the function_space,
		so that it could also be used as DirichletBC(function_space, value, ...).
	boundaries : dolfin.cpp.mesh.MeshFunctionSizet
		The MeshFunction object representing the boundaries.
	idcs : list[int]
		A list of indices / boundary markers that determine the boundaries
		onto which the Dirichlet boundary conditions should be applied to.

	Returns
	-------
	list[dolfin.fem.dirichletbc.DirichletBC]
		A list of DirichletBC objects that represent the boundary conditions.
	"""

	bcs_list = []
	for i in idcs:
		bcs_list.append(fenics.DirichletBC(function_space, value, boundaries, i))

	return bcs_list



class Interpolator:
	"""Efficient interpolation between two function spaces.

	This is very useful, if multiple interpolations have to be
	carried out between the same spaces, which is made significantly
	faster by computing the corresponding matrix.
	The function spaces can even be defined on different meshes.
	"""

	def __init__(self, V, W):
		"""Initializes the object.

		Parameters
		----------
		V : dolfin.function.functionspace.FunctionSpace
			The function space whose objects shall be interpolated.
		W : dolfin.function.functionspace.FunctionSpace
			The space into which they shall be interpolated.
		"""

		self.V = V
		self.W = W
		self.transfer_matrix = fenics.PETScDMCollection.create_transfer_matrix(self.V, self.W)


	def interpolate(self, u):
		"""Interpolates function to target space.

		Parameters
		----------
		u : dolfin.function.function.Function
			The function that shall be interpolated.

		Returns
		-------
		dolfin.function.function.Function
			The result of the interpolation.
		"""

		assert u.function_space() == self.V, 'input does not belong to the correct function space'
		v = fenics.Function(self.W)
		v.vector()[:] = (self.transfer_matrix*u.vector())[:]

		return v



def _assemble_petsc_system(A_form, b_form, bcs=None):
	"""Assembles a system symmetrically and converts to PETSc

	Also uses ident_zeros() to get well-posed problems

	Parameters
	----------
	A_form : ufl.form.Form
		the UFL form for the LHS
	b_form : ufl.form.Form
		the UFL form for the RHS
	bcs : None or dolfin.fem.dirichletbc.DirichletBC or list[dolfin.fem.dirichletbc.DirichletBC]
		list of Dirichlet boundary conditions

	Returns
	-------
	petsc4py.PETSc.Mat
		the petsc matrix for the LHS
	petsc4py.PETSc.Vec
		the petsc matrix for the RHS
	"""

	A, b = fenics.assemble_system(A_form, b_form, bcs, keep_diagonal=True)
	A.ident_zeros()

	A = fenics.as_backend_type(A).mat()
	b = fenics.as_backend_type(b).vec()

	return A, b



def _setup_petsc_options(ksps, ksp_options):
	"""Sets up an (iterative) linear solver.

	This is used to pass user defined command line options for PETSc
	to the PETSc KSP objects.
	Here, options[i] is applied to ksps[i]

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

	assert len(ksps) == len(ksp_options), 'Length of options and ksps does not match'

	opts = fenics.PETScOptions

	for i in range(len(ksps)):
		opts.clear()

		for option in ksp_options[i]:
			opts.set(*option)

		ksps[i].setFromOptions()
