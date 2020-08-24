"""Module including utility and helper functions.

These module includes utility and helper functions used in cestrel. They
might also be interesting for users, so they are part of the public API.
They include wrappers that allow to shorten the coding for often recurring
actions.
"""

import fenics
import configparser
import numpy as np
from ._exceptions import PETScKSPError, InputError
from petsc4py import PETSc



def summation(x):
	"""Sums elements of a list in a UFL friendly fashion.

	This can be used to sum, e.g., UFL forms, or UFL expressions
	that can be used in UFL forms.

	Parameters
	----------
	x : list[ufl.form.Form] or list[int] or list[float]
		The list of entries that shall be summed.

	Returns
	-------
	y : ufl.form.Form or int or float
		Sum of input (same type as entries of input).

	See Also
	--------
	multiplication : Multiplies the elements of a list.

	Notes
	-----
	For "usual" summation of integers or floats, the built-in sum function
	of python or the numpy variant are recommended. Still, they are
	incompatible with fenics objects, so this function should be used for
	the latter.

	Examples
	--------
	    a = cestrel.summation([u.dx(i)*v.dx(i)*dx for i in mesh.geometric_dimension()])

	is equivalent to

	    a = u.dx(0)*v.dx(0)*dx + u.dx(1)*v.dx(1)*dx

	(for a 2D mesh).
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
	"""Multiplies the elements of a list in a UFL friendly fashion.

	Used to build the product of certain UFL expressions to construct
	a UFL form.

	Parameters
	----------
	x : list[ufl.core.expr.Expr] or list[int] or list[float]
		The list whose entries shall be multiplied.

	Returns
	-------
	y : ufl.core.expr.Expr or int or float
		The result of the multiplication.

	See Also
	--------
	summation : Sums elements of a list.

	Examples
	--------
	    a = cestrel.multiplication([u.dx(i) for i in range(mesh.geometric_dimension())])

	is equivalent to

	    a = u.dx(0) * u.dx(1)

	(for a 2D mesh).
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

	Examples
	--------
	    dm = EmptyMeasure(dx)
	    u*dm

	is equivalent to

	    Constant(0)*u*dm

	so that this generates zeros when assembled over.
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
		"""Multiplies the empty measure to the right.

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
	ufl.measure.Measure or cestrel.utils.EmptyMeasure
		The corresponding sum of the measures or an empty measure.

	Examples
	--------
	    from fenics import *
	    import cestrel
	    mesh, _, boundaries, dx, ds, _ = cestrel.regular_mesh(25)
	    top_bottom_measure = cestrel.utils.generate_measure([3,4], ds)
	    assemble(1*top_bottom_measure)
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
	configparser package.

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
	idcs : list[int] or int
		A list of indices / boundary markers that determine the boundaries
		onto which the Dirichlet boundary conditions should be applied to.
		Can also be a single integer for a single boundary.

	Returns
	-------
	list[dolfin.fem.dirichletbc.DirichletBC]
		A list of DirichletBC objects that represent the boundary conditions.

	Examples
	--------
	Generate homogeneous Dirichlet boundary conditions for all 4 sides of the unit square.

	    from fenics import *
	    import cestrel

	    mesh, _, _, _, _, _ = cestrel.regular_mesh(25)
	    V = FunctionSpace(mesh, 'CG', 1)
	    bcs = cestrel.create_bcs_list(V, Constant(0), boundaries, [1,2,3,4])
	"""

	bcs_list = []
	if type(idcs) == list:
		for i in idcs:
			bcs_list.append(fenics.DirichletBC(function_space, value, boundaries, i))

	elif type(idcs) == int:
		bcs_list.append(fenics.DirichletBC(function_space, value, boundaries, idcs))

	return bcs_list





class Interpolator:
	"""Efficient interpolation between two function spaces.

	This is very useful, if multiple interpolations have to be
	carried out between the same spaces, which is made significantly
	faster by computing the corresponding matrix.
	The function spaces can even be defined on different meshes.

	Examples
	--------
	    from fenics import *
	    import cestrel

	    mesh, _, _, _, _, _ = cestrel.regular_mesh(25)
	    V1 = FunctionSpace(mesh, 'CG', 1)
	    V2 = FunctionSpace(mesh, 'CG', 2)

	    expr = Expression('sin(2*pi*x[0])', degree=1)
	    u = interpolate(expr, V1)

	    interp = cestrel.utils.Interpolator(V1, V2)
	    interp.interpolate(u)
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

		The function has to belong to the origin space, i.e., the first argument
		of __init__, and it is interpolated to the destination space, i.e., the
		second argument of __init__. There is no need to call set_allow_extrapolation
		on the function (this is done automatically due to the method).

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
	"""Assembles a system symmetrically and converts objects to PETSc format.

	Parameters
	----------
	A_form : ufl.form.Form
		The UFL form for the left-hand side of the linear equation.
	b_form : ufl.form.Form
		The UFL form for the right-hand side of the linear equation.
	bcs : None or dolfin.fem.dirichletbc.DirichletBC or list[dolfin.fem.dirichletbc.DirichletBC]
		A list of Dirichlet boundary conditions.

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

	A, b = fenics.assemble_system(A_form, b_form, bcs, keep_diagonal=True)
	A.ident_zeros()

	A = fenics.as_backend_type(A).mat()
	b = fenics.as_backend_type(b).vec()

	return A, b



def _setup_petsc_options(ksps, ksp_options):
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

	assert len(ksps) == len(ksp_options), 'Length of options and ksps does not match'

	opts = fenics.PETScOptions

	for i in range(len(ksps)):
		opts.clear()

		for option in ksp_options[i]:
			opts.set(*option)

		ksps[i].setFromOptions()



def _solve_linear_problem(ksp=None, A=None, b=None, x=None):
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

	Returns
	-------
	petsc4py.PETSc.Vec
		The solution vector.
	"""

	if ksp is None:
		ksp = PETSc.KSP().create()
		options = [[
			['ksp_type', 'preonly'],
			['pc_type', 'lu'],
			['pc_factor_mat_solver_type', 'mumps'],
			['mat_mumps_icntl_24', 1]
		]]

		_setup_petsc_options([ksp], options)


	if A is not None:
		ksp.setOperators(A)
	else:
		A = ksp.getOperators()[0]
		if A.size[0] == -1 and A.size[1] == -1:
			raise InputError('The KSP object has to be initialized with some Matrix in case A is None.')

	if b is None:
		return A.getVecs()[0]

	if x is None:
		x, _ = A.getVecs()

	ksp.solve(b, x)

	if ksp.getConvergedReason() < 0:
		raise PETScKSPError(ksp.getConvergedReason())

	return x



def write_out_mesh(mesh, original_msh_file, out_msh_file):
	"""Writes out the current mesh as .msh file.

	This method updates the vertex positions in the `original_gmsh_file`, the
	topology of the mesh and its connections are the same. The original gmsh
	file is kept, and a new one is generated under `out_mesh_file`.

	Parameters
	----------
	mesh : dolfin.cpp.mesh.Mesh
		The mesh object in fenics that should be saved as gmsh file.
	original_msh_file : str
		Path to the original gmsh mesh file of the mesh object, has to
		end with '.msh'.
	out_msh_file : str
		Path (and name) of the output mesh file, has to end with '.msh'.

	Returns
	-------
	None

	Notes
	-----
	The method only works with gmsh mesh 4.1 file format. Others might also work,
	but this is not tested or ensured in any way.
	"""

	assert original_msh_file[-4:] == '.msh', 'Format for original_mesh_file is wrong'
	assert out_msh_file[-4:] == '.msh', 'Format for out_mesh_file is wrong'

	dim = mesh.geometric_dimension()

	with open(original_msh_file, 'r') as old_file, open(out_msh_file, 'w') as new_file:

		points = mesh.coordinates()

		node_section = False
		info_section = False
		subnode_counter = 0
		subwrite_counter = 0
		idcs = np.zeros(1, dtype=int)

		for line in old_file:
			if line == '$EndNodes\n':
				node_section = False

			if not node_section:
				new_file.write(line)
			else:
				split_line = line.split(' ')
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
							mod_line = format(points[idcs[subwrite_counter]][0], '.16f') + ' ' + format(points[idcs[subwrite_counter]][1], '.16f') + ' ' + '0\n'
						elif dim == 3:
							mod_line = format(points[idcs[subwrite_counter]][0], '.16f') + ' ' + format(points[idcs[subwrite_counter]][1], '.16f') + ' ' + format(points[idcs[subwrite_counter]][2], '.16f') + '\n'
						else:
							raise InputError('Not a valid dimension for the mesh.')
						new_file.write(mod_line)
						subwrite_counter += 1


			if line == '$Nodes\n':
				node_section = True
				info_section = True
