"""
Created on 1/30/19, 2:27 PM

@author: blauths
"""

import fenics
import numpy as np
import time
from petsc4py import PETSc



def MeshGen(mesh_file):
	"""
	Imports a mesh file in .xdmf format and generates all necessary geometrical information for fenics

	Parameters
	----------
	mesh_file : str
		location of the "main" mesh file in .xdmf file format

	Returns
	-------
	mesh : dolfin.cpp.mesh.Mesh
		The computational mesh
	subdomains : dolfin.cpp.mesh.MeshFunctionSizet
		A MeshFunction containing the subdomains, i.e., the physical regions of dimension d generated in gmsh
	boundaries : dolfin.cpp.mesh.MeshFunctionSizet
		A MeshFunction containing the boundaries, i.e., the physical regions of dimension d-1 generated in gmsh
	dx : ufl.measure.Measure
		The volume measure corresponding to the mesh and subdomains
	ds : ufl.measure.Measure
		The surface measure corresponding to the mesh and boundaries
	dS : ufl.measure.Measure
		The interior face measure corresponding to the mesh and boundaries
	"""
	
	start_time = time.time()
	print('Importing mesh to FEniCS')
	# Check for the file format
	if mesh_file[-5:] == '.xdmf':
		file_string = mesh_file[:-5]
	else:
		raise SystemExit('Not a suitable mesh file format')
	
	mesh = fenics.Mesh()
	xdmf_file = fenics.XDMFFile(mesh.mpi_comm(), mesh_file)
	xdmf_file.read(mesh)
	
	subdomains_mvc = fenics.MeshValueCollection('size_t', mesh, mesh.geometric_dimension())
	boundaries_mvc = fenics.MeshValueCollection('size_t', mesh, mesh.geometric_dimension() - 1)
	
	xdmf_subdomains = fenics.XDMFFile(mesh.mpi_comm(), file_string + '_subdomains.xdmf')
	xdmf_boundaries = fenics.XDMFFile(mesh.mpi_comm(), file_string + '_boundaries.xdmf')
	
	xdmf_boundaries.read(boundaries_mvc, 'boundaries')
	xdmf_subdomains.read(subdomains_mvc, 'subdomains')
	
	subdomains = fenics.MeshFunction('size_t', mesh, subdomains_mvc)
	boundaries = fenics.MeshFunction('size_t', mesh, boundaries_mvc)
	
	xdmf_file.close()
	xdmf_subdomains.close()
	xdmf_boundaries.close()
	
	dx = fenics.Measure('dx', domain=mesh, subdomain_data=subdomains)
	ds = fenics.Measure('ds', domain=mesh, subdomain_data=boundaries)
	dS = fenics.Measure('dS', domain=mesh, subdomain_data=boundaries)
	
	end_time = time.time()
	print('Done Importing Mesh. Elapsed Time: ' + format(end_time - start_time, '.3e') + ' s')
	print('')
	
	return mesh, subdomains, boundaries, dx, ds, dS



def regular_mesh(n=10, lx=1.0, ly=1.0, lz=None):
	"""Creates a regular mesh in either 2D (rectangle) or 3D (cube), starting at the origin and having specified lengths

	Parameters
	----------
	n : int
		Number of subdivisions in the smallest coordinate direction
	lx : float
		length in x-direction
	ly : float
		length in y-direction
	lz : float or None
		length in z-direction (geometry will be 2D if lz==None)

	Returns
	-------
	mesh : dolfin.cpp.mesh.Mesh
		The computational mesh
	subdomains : dolfin.cpp.mesh.MeshFunctionSizet
		A MeshFunction containing the subdomains
	boundaries : dolfin.cpp.mesh.MeshFunctionSizet
		A MeshFunction containing the boundaries. Marker 1 corresponds to x=0, marker 2 to x=lx, marker 3 to y=0 and so on
	dx : ufl.measure.Measure
		The volume measure corresponding to the mesh and subdomains
	ds : ufl.measure.Measure
		The surface measure corresponding to the mesh and boundaries
	dS : ufl.measure.Measure
		The interior face measure corresponding to the mesh and boundaries
	"""

	n = int(n)
	
	if lz is None:
		sizes = [lx, ly]
		dim = 2
	else:
		sizes = [lx, ly, lz]
		dim = 3
	
	size_min = np.min(sizes)
	num_points = [int(np.round(length/size_min*n)) for length in sizes]
	
	if lz is None:
		mesh = fenics.RectangleMesh(fenics.Point(0, 0), fenics.Point(sizes), num_points[0], num_points[1])
	else:
		mesh = fenics.BoxMesh(fenics.Point(0, 0, 0), fenics.Point(sizes), num_points[0], num_points[1], num_points[2])
	
	subdomains = fenics.MeshFunction('size_t', mesh, dim=dim)
	boundaries = fenics.MeshFunction('size_t', mesh, dim=dim - 1)
	
	x_min = fenics.CompiledSubDomain('on_boundary && near(x[0], 0, tol)', tol=fenics.DOLFIN_EPS)
	x_max = fenics.CompiledSubDomain('on_boundary && near(x[0], length, tol)', tol=fenics.DOLFIN_EPS, length=sizes[0])
	
	y_min = fenics.CompiledSubDomain('on_boundary && near(x[1], 0, tol)', tol=fenics.DOLFIN_EPS)
	y_max = fenics.CompiledSubDomain('on_boundary && near(x[1], length, tol)', tol=fenics.DOLFIN_EPS, length=sizes[1])
	
	if lz is not None:
		z_min = fenics.CompiledSubDomain('on_boundary && near(x[2], 0, tol)', tol=fenics.DOLFIN_EPS)
		z_max = fenics.CompiledSubDomain('on_boundary && near(x[2], length, tol)', tol=fenics.DOLFIN_EPS, length=sizes[2])
	
	x_min.mark(boundaries, 1)
	x_max.mark(boundaries, 2)
	y_min.mark(boundaries, 3)
	y_max.mark(boundaries, 4)
	
	if lz is not None:
		z_min.mark(boundaries, 5)
		z_max.mark(boundaries, 6)
	
	dx = fenics.Measure('dx', mesh, subdomain_data=subdomains)
	ds = fenics.Measure('ds', mesh, subdomain_data=boundaries)
	dS = fenics.Measure('dS', mesh)
	
	return mesh, subdomains, boundaries, dx, ds, dS


def regular_box_mesh(n=10, sx=0.0, sy=0.0, sz=None, ex=1.0, ey=1.0, ez=None):
	"""Creates a regular box mesh of the rectangle [sx, ex] x [sy, ey] or the cube [sx, ex] x [sy, ey] x [sz, ez]

	Parameters
	----------
	n : int
		Number of subdivisions for the smallest direction
	sx : float
		Start of the x-interval
	sy : float
		Start of the y-interval
	sz : float or None
		Start of the z-interval
	ex : float
		End of the x-interval
	ey : float
		End of the y-interval
	ez : float or None
		End of the z-interval

	Returns
	-------
	mesh : dolfin.cpp.mesh.Mesh
		The computational mesh
	subdomains : dolfin.cpp.mesh.MeshFunctionSizet
		A MeshFunction containing the subdomains
	boundaries : dolfin.cpp.mesh.MeshFunctionSizet
		A MeshFunction containing the boundaries. Marker 1 corresponds to x=sx, marker 2 to x=ex, marker 3 to y=sy and so on
	dx : ufl.measure.Measure
		The volume measure corresponding to the mesh and subdomains
	ds : ufl.measure.Measure
		The surface measure corresponding to the mesh and boundaries
	dS : ufl.measure.Measure
		The interior face measure corresponding to the mesh and boundaries
	"""

	n = int(n)

	assert sx < ex, 'Incorrect input for the x-coordinate'
	assert sy < ey, 'Incorrect input for the y-coordinate'
	assert (sz is None and ez is None) or (sz < ez), 'Incorrect input for the z-coordinate'

	if sz is None:
		lx = ex - sx
		ly = ey - sy
		sizes = [lx, ly]
		dim = 2
	else:
		lx = ex - sx
		ly = ey - sy
		lz = ez - sz
		sizes = [lx, ly, lz]
		dim = 3

	size_min = np.min(sizes)
	num_points = [int(np.round(length/size_min*n)) for length in sizes]

	if sz is None:
		mesh = fenics.RectangleMesh(fenics.Point(sx, sy), fenics.Point(ex, ey), num_points[0], num_points[1])
	else:
		mesh = fenics.BoxMesh(fenics.Point(sx, sy, sz), fenics.Point(ex, ey, ez), num_points[0], num_points[1], num_points[2])

	subdomains = fenics.MeshFunction('size_t', mesh, dim=dim)
	boundaries = fenics.MeshFunction('size_t', mesh, dim=dim - 1)

	x_min = fenics.CompiledSubDomain('on_boundary && near(x[0], sx, tol)', tol=fenics.DOLFIN_EPS, sx=sx)
	x_max = fenics.CompiledSubDomain('on_boundary && near(x[0], ex, tol)', tol=fenics.DOLFIN_EPS, ex=ex)

	y_min = fenics.CompiledSubDomain('on_boundary && near(x[1], sy, tol)', tol=fenics.DOLFIN_EPS, sy=sy)
	y_max = fenics.CompiledSubDomain('on_boundary && near(x[1], ey, tol)', tol=fenics.DOLFIN_EPS, ey=ey)

	if sz is not None:
		z_min = fenics.CompiledSubDomain('on_boundary && near(x[2], sz, tol)', tol=fenics.DOLFIN_EPS, sz=sz)
		z_max = fenics.CompiledSubDomain('on_boundary && near(x[2], ez, tol)', tol=fenics.DOLFIN_EPS, ez=ez)

	x_min.mark(boundaries, 1)
	x_max.mark(boundaries, 2)
	y_min.mark(boundaries, 3)
	y_max.mark(boundaries, 4)

	if sz is not None:
		z_min.mark(boundaries, 5)
		z_max.mark(boundaries, 6)

	dx = fenics.Measure('dx', mesh, subdomain_data=subdomains)
	ds = fenics.Measure('ds', mesh, subdomain_data=boundaries)
	dS = fenics.Measure('dS', mesh)

	return mesh, subdomains, boundaries, dx, ds, dS





class MeshHandler:
	def __init__(self, shape_form_handler):
		"""

		Parameters
		----------
		shape_form_handler : adpack.forms.ShapeFormHandler
		"""

		self.shape_form_handler = shape_form_handler
		# Namespacing
		self.mesh = self.shape_form_handler.mesh
		self.dx = self.shape_form_handler.dx
		self.bbtree = self.mesh.bounding_box_tree()
		self.config = self.shape_form_handler.config

		self.check_a_priori = self.config.getboolean('MeshQuality', 'check_a_priori')
		self.check_a_posteriori = self.config.getboolean('MeshQuality', 'check_a_posteriori')

		self.radius_ratios_initial_mf = fenics.MeshQuality.radius_ratios(self.mesh)
		self.radius_ratios_initial = self.radius_ratios_initial_mf.array().copy()

		self.mesh_quality_tol = self.config.getfloat('MeshQuality', 'qtol')
		self.min_quality = 1.0



	def move_mesh(self, transformation):
		"""
		Move the mesh according to the diffeomorphism id + transformation

		Parameters
		----------
		transformation : dolfin.function.function.Function
			The transformation for the mesh
		"""

		assert transformation.ufl_element().family() == 'Lagrange' and transformation.ufl_element().degree() == 1, 'Not a valid mesh transformation'

		if not self.test_a_priori(transformation):
			return False
		else:
			self.old_coordinates = self.mesh.coordinates().copy()
			fenics.ALE.move(self.mesh, transformation)
			self.bbtree.build(self.mesh)

			return self.test_a_posteriori()



	def revert_transformation(self):
		"""
		Reverts the previous transformation done in self.move_mesh
		"""

		self.mesh.coordinates()[:, :] = self.old_coordinates
		self.bbtree.build(self.mesh)



	def compute_decreases(self, search_direction, stepsize):
		"""
		Gives a better estimation of the stepsize. The output is the number of "Armijo halvings" we have to do in order to
		get a transformation that satisfies norm(transformation)_fro <= 0.3, where transformation = stepsize*search_direction.
		Due to the linearity of the norm this has to be done only once, all smaller stepsizes are feasible wrt. to this criterion as well

		Parameters
		----------
		search_direction : dolfin.function.function.Function
			The search direction in the optimization routine / descent algorithm
		stepsize : float
			The stepsize in the descent algorithm

		Returns
		 : float
			A guess for the number of "Armijo halvings" to get a better stepsize
		-------

		"""

		angle_change = float(self.config.get('MeshQuality', 'angle_change'))

		assert angle_change > 0, 'Angle change has to be positive'
		if angle_change == float('inf'):
			return 0

		else:
			opts = fenics.PETScOptions
			opts.clear()
			opts.set('ksp_type', 'preonly')
			opts.set('pc_type', 'jacobi')
			opts.set('pc_jacobi_type', 'diagonal')
			opts.set('ksp_rtol', 1e-16)
			opts.set('ksp_atol', 1e-20)
			opts.set('ksp_max_it', 1000)

			DG0 = self.shape_form_handler.DG0
			v = fenics.TrialFunction(DG0)
			w = fenics.TestFunction(DG0)

			a = v*w*self.dx
			L_norm = fenics.sqrt(fenics.inner(fenics.grad(search_direction), fenics.grad(search_direction)))*w*self.dx

			A = fenics.assemble(a)
			b_norm = fenics.assemble(L_norm)
			A = fenics.as_backend_type(A).mat()
			b_norm = fenics.as_backend_type(b_norm).vec()
			x_norm, _ = A.getVecs()

			ksp = PETSc.KSP().create()
			ksp.setFromOptions()
			ksp.setOperators(A)
			ksp.setUp()
			ksp.solve(b_norm, x_norm)
			if ksp.getConvergedReason() < 0:
				raise SystemExit('Krylov solver did not converge. Reason: ' + str(ksp.getConvergedReason()))

			frobenius_norm = np.max(x_norm[:])

			beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo')

			return np.maximum(np.ceil(np.log(angle_change/stepsize/frobenius_norm)/np.log(1/beta_armijo)), 0.0)



	def test_a_priori(self, transformation):
		"""
		Checks the quality of the transformation. The criterion is that det(I + D transformation) should neither be too large nor too small
		in order to achieve the best transformations

		Parameters
		----------
		transformation : dolfin.function.function.Function
			The transformation for the mesh

		Returns
		 : bool
			A boolean that indicates whether the desired transformation is feasible
		-------

		"""

		if self.check_a_priori:

			opts = fenics.PETScOptions
			opts.clear()
			opts.set('ksp_type', 'preonly')
			opts.set('pc_type', 'jacobi')
			opts.set('pc_jacobi_type', 'diagonal')
			opts.set('ksp_rtol', 1e-16)
			opts.set('ksp_atol', 1e-20)
			opts.set('ksp_max_it', 1000)

			dim = self.mesh.geometric_dimension()
			DG0 = self.shape_form_handler.DG0
			v = fenics.TrialFunction(DG0)
			w = fenics.TestFunction(DG0)
			volume_change = float(self.config.get('MeshQuality', 'volume_change'))

			assert volume_change > 1, 'Volume change has to be larger than 1'

			a = v*w*self.dx
			L = fenics.det(fenics.Identity(dim) + fenics.grad(transformation))*w*self.dx

			A = fenics.assemble(a)
			b = fenics.assemble(L)
			A = fenics.as_backend_type(A).mat()
			b = fenics.as_backend_type(b).vec()
			x, _ = A.getVecs()

			ksp = PETSc.KSP().create()
			ksp.setFromOptions()
			ksp.setOperators(A)
			ksp.setUp()
			ksp.solve(b, x)
			if ksp.getConvergedReason() < 0:
				raise SystemExit('Krylov solver did not converge. Reason: ' + str(ksp.getConvergedReason()))

			min_det = np.min(x[:])
			max_det = np.max(x[:])

			return (min_det >= 1/volume_change) and (max_det <= volume_change)

		else:
			return True



	def test_a_posteriori(self):
		"""
		Checks whether the mesh is a valid finite element mesh after it has been moved (fenics accepts overlapping elements by default)
		"""

		if self.check_a_posteriori:
			mesh = self.mesh
			cells = mesh.cells()
			coordinates = mesh.coordinates()
			self_intersections = False
			for i in range(coordinates.shape[0]):
				x = fenics.Point(coordinates[i])
				cells_idx = self.bbtree.compute_entity_collisions(x)
				intersections = len(cells_idx)
				M = cells[cells_idx]
				occurences = M.flatten().tolist().count(i)

				if intersections > occurences:
					self_intersections = True
					break

			if self_intersections:
				self.revert_transformation()
				return False
			else:
				return True

		else:
			return True



	def compute_relative_quality(self):
		radius_ratios_mf = fenics.MeshQuality.radius_ratios(self.mesh)
		self.radius_ratios = radius_ratios_mf.array().copy()
		relative_quality = self.radius_ratios / self.radius_ratios_initial
		self.min_quality = np.min(relative_quality)


	def remesh(self):
		"""
		A remeshing routine, that is called when the mesh quality is too bad.
		THIS IS NOT IMPLEMENTED YET

		Returns
		-------

		"""
		# TODO: Implement Remeshing
		raise NotImplementedError('The remeshing routine is not yet implemented')
