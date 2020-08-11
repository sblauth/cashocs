"""
Created on 1/30/19, 2:27 PM

@author: blauths
"""

import fenics
import numpy as np
import time
from petsc4py import PETSc
import os
import sys
import uuid



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
	xdmf_file.close()
	
	subdomains_mvc = fenics.MeshValueCollection('size_t', mesh, mesh.geometric_dimension())
	boundaries_mvc = fenics.MeshValueCollection('size_t', mesh, mesh.geometric_dimension() - 1)

	if os.path.exists(file_string + '_subdomains.xdmf'):
		xdmf_subdomains = fenics.XDMFFile(mesh.mpi_comm(), file_string + '_subdomains.xdmf')
		xdmf_subdomains.read(subdomains_mvc, 'subdomains')
		xdmf_subdomains.close()
	if os.path.exists(file_string + '_boundaries.xdmf'):
		xdmf_boundaries = fenics.XDMFFile(mesh.mpi_comm(), file_string + '_boundaries.xdmf')
		xdmf_boundaries.read(boundaries_mvc, 'boundaries')
		xdmf_boundaries.close()

	subdomains = fenics.MeshFunction('size_t', mesh, subdomains_mvc)
	boundaries = fenics.MeshFunction('size_t', mesh, boundaries_mvc)

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
	x_min.mark(boundaries, 1)
	x_max.mark(boundaries, 2)

	y_min = fenics.CompiledSubDomain('on_boundary && near(x[1], 0, tol)', tol=fenics.DOLFIN_EPS)
	y_max = fenics.CompiledSubDomain('on_boundary && near(x[1], length, tol)', tol=fenics.DOLFIN_EPS, length=sizes[1])
	y_min.mark(boundaries, 3)
	y_max.mark(boundaries, 4)

	if lz is not None:
		z_min = fenics.CompiledSubDomain('on_boundary && near(x[2], 0, tol)', tol=fenics.DOLFIN_EPS)
		z_max = fenics.CompiledSubDomain('on_boundary && near(x[2], length, tol)', tol=fenics.DOLFIN_EPS, length=sizes[2])
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
	x_min.mark(boundaries, 1)
	x_max.mark(boundaries, 2)

	y_min = fenics.CompiledSubDomain('on_boundary && near(x[1], sy, tol)', tol=fenics.DOLFIN_EPS, sy=sy)
	y_max = fenics.CompiledSubDomain('on_boundary && near(x[1], ey, tol)', tol=fenics.DOLFIN_EPS, ey=ey)
	y_min.mark(boundaries, 3)
	y_max.mark(boundaries, 4)

	if sz is not None:
		z_min = fenics.CompiledSubDomain('on_boundary && near(x[2], sz, tol)', tol=fenics.DOLFIN_EPS, sz=sz)
		z_max = fenics.CompiledSubDomain('on_boundary && near(x[2], ez, tol)', tol=fenics.DOLFIN_EPS, ez=ez)
		z_min.mark(boundaries, 5)
		z_max.mark(boundaries, 6)

	dx = fenics.Measure('dx', mesh, subdomain_data=subdomains)
	ds = fenics.Measure('ds', mesh, subdomain_data=boundaries)
	dS = fenics.Measure('dS', mesh)

	return mesh, subdomains, boundaries, dx, ds, dS





class MeshHandler:
	"""This class implements all mesh related things for the shape optimization, such as transformations and remeshing

	Attributes
	----------
	shape_optimization_problem : adpack.shape_optimization.shape_optimization_problem.ShapeOptimizationProblem
		The corresponding shape optimization problem

	shape_form_handler : adpack.forms.ShapeFormHandler
		The shape form handler of the problem

	mesh : dolfin.cpp.mesh.Mesh
		The finite element mesh

	dx : ufl.measure.Measure
		The volume measure corresponding to mesh

	bbtree : dolfin.cpp.geometry.BoundingBoxTree
		Bounding box tree for the mesh, needs to be updated after moving the mesh

	config : configparser.ConfigParser
		config object for the shape optimization problem

	check_a_priori : bool
		boolean flag, en- or disabling checking the quality of the shape deformation (before the actual mesh is moved)

	check_a_posteriori : bool
		boolean flag, en- or disabling checking the quality of the shape deformation (after the actual mesh is moved)

	radius_ratios_initial_mf : dolfin.cpp.mesh.MeshFunctionDouble
		MeshFunction representing the radius ratios for the initial geometry

	radius_ratios_initial : numpy.ndarray
		numpy array of the radius ratios for the initial geometry

	mesh_quality_tol : float
		relative threshold for remeshing

	min_quality : float
		measures the minimal (relative) mesh quality regarding the radius ratios

	old_coordinates : numpy.ndarray
		numpy array of coordinates of the mesh's vertices, used for reverting the deformation

	radius_ratios : numpy.ndarray
		numpy array of the radius ratios for the current geometry

	temp_file : str
		temporary path to a .msh file where the current mesh is written out to

	new_gmsh_file : str
		path to the new (remeshed) .msh file

	new_xdmf_file : str
		path to the remeshed mesh, in xdmf format
	"""

	def __init__(self, shape_optimization_problem):
		"""Initializes the MeshHandler object

		Parameters
		----------
		shape_optimization_problem : adpack.shape_optimization.shape_optimization_problem.ShapeOptimizationProblem
		"""

		self.shape_optimization_problem = shape_optimization_problem
		self.shape_form_handler = self.shape_optimization_problem.shape_form_handler
		# Namespacing
		self.mesh = self.shape_form_handler.mesh
		self.dx = self.shape_form_handler.dx
		self.bbtree = self.mesh.bounding_box_tree()
		self.config = self.shape_form_handler.config

		self.check_a_priori = self.config.getboolean('MeshQuality', 'check_a_priori', fallback=True)
		self.check_a_posteriori = self.config.getboolean('MeshQuality', 'check_a_posteriori', fallback=True)

		self.radius_ratios_initial_mf = fenics.MeshQuality.radius_ratios(self.mesh)
		self.radius_ratios_initial = self.radius_ratios_initial_mf.array().copy()

		self.mesh_quality_tol = self.config.getfloat('MeshQuality', 'qtol', fallback=0.25)
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

		if not self.__test_a_priori(transformation):
			return False
		else:
			self.old_coordinates = self.mesh.coordinates().copy()
			fenics.ALE.move(self.mesh, transformation)
			self.bbtree.build(self.mesh)

			return self.__test_a_posteriori()



	def revert_transformation(self):
		"""Reverts the previous transformation done in self.move_mesh, and restores the mesh.coordinates to old_coordinates

		Returns
		-------
		None
		"""

		self.mesh.coordinates()[:, :] = self.old_coordinates
		self.bbtree.build(self.mesh)



	def compute_decreases(self, search_direction, stepsize):
		"""Computes the number of decreases needed in the Armijo rule so that the Frobenius norm criterion is satisfied

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
		-------
		int
			A guess for the number of "Armijo halvings" to get a better stepsize
		"""

		angle_change = float(self.config.get('MeshQuality', 'angle_change', fallback='inf'))

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

			beta_armijo = self.config.getfloat('OptimizationRoutine', 'beta_armijo', fallback=2)

			return np.maximum(np.ceil(np.log(angle_change/stepsize/frobenius_norm)/np.log(1/beta_armijo)), 0.0)



	def __test_a_priori(self, transformation):
		"""Check the quality of the transformation before the actual mesh is moved

		Checks the quality of the transformation. The criterion is that det(I + D transformation) should neither be too large nor too small
		in order to achieve the best transformations

		Parameters
		----------
		transformation : dolfin.function.function.Function
			The transformation for the mesh

		Returns
		-------
		bool
			A boolean that indicates whether the desired transformation is feasible
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
			volume_change = float(self.config.get('MeshQuality', 'volume_change', fallback='inf'))

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



	def __test_a_posteriori(self):
		"""Checks whether the mesh is a valid finite element mesh after it has been moved (fenics accepts overlapping elements by default)

		Returns
		-------
		bool
			True if the test is successful, False otherwise
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
		"""Computes the relative quality (based on radius ratios) for the current mesh

		Returns
		-------
		None
		"""

		radius_ratios_mf = fenics.MeshQuality.radius_ratios(self.mesh)
		self.radius_ratios = radius_ratios_mf.array().copy()
		relative_quality = self.radius_ratios / self.radius_ratios_initial
		self.min_quality = np.min(relative_quality)



	def write_out_mesh(self):
		"""Writes out the current mesh as .msh file

		Returns
		-------
		None
		"""

		# TODO: Put this as a general, accessible routine

		dim = self.mesh.geometric_dimension()

		if self.shape_form_handler.remesh_counter == 0:
			old_file = open(self.shape_form_handler.gmsh_file, 'r')
		else:
			old_file = open(self.shape_form_handler.gmsh_file[:-4] + '_' + self.shape_form_handler.remesh_counter + '.msh', 'r')
		self.temp_file = self.shape_form_handler.remesh_directory + '/mesh_' + str(uuid.uuid4().hex) + '.msh'
		new_file = open(self.temp_file, 'w')

		points = self.mesh.coordinates()

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
						new_file.write(mod_line)
						subwrite_counter += 1


			if line == '$Nodes\n':
				node_section = True
				info_section = True

		old_file.close()
		new_file.close()



	def __generate_remesh_geo(self):
		"""Generates a .geo file used for remeshing

		Returns
		-------
		None
		"""

		with open(self.shape_form_handler.remesh_geo_file, 'w') as file:
			temp_name = os.path.split(self.temp_file)[1]
			file.write('Merge \'' + temp_name + '\';\n')
			file.write('CreateGeometry;\n')
			file.write('\n')

			geo_file = self.config.get('Mesh', 'geo_file')
			with open(geo_file, 'r') as f:
				for line in f:
					if line[:2] == 'lc':
						file.write(line)
					if line[:5] == 'Field':
						file.write(line)
					if line[:16] == 'Background Field':
						file.write(line)



	def remesh(self):
		"""Performs a remeshing of the geometry, and then restarts the optimization problem with the new mesh

		Returns
		-------
		None
		"""

		if self.shape_form_handler.do_remesh:
			self.write_out_mesh()
			self.__generate_remesh_geo()

			dim = self.mesh.geometric_dimension()

			gmsh_command = 'gmsh ' + self.shape_form_handler.remesh_geo_file + ' -' + str(int(dim)) + ' -o ' + self.temp_file
			# os.system(gmsh_command + ' >/dev/null 2>&1')
			os.system(gmsh_command)
			self.shape_form_handler.remesh_counter += 1
			self.config.set('Mesh', 'remesh_counter', str(self.shape_form_handler.remesh_counter))

			self.new_gmsh_file = self.shape_form_handler.remesh_directory + '/mesh_' + str(self.shape_form_handler.remesh_counter) + '.msh'
			rename_command = 'mv ' + self.temp_file + ' ' + self.new_gmsh_file
			os.system(rename_command)

			self.new_xdmf_file = self.shape_form_handler.remesh_directory + '/mesh_' + str(self.shape_form_handler.remesh_counter) + '.xdmf'
			convert_command = 'mesh-convert ' + self.new_gmsh_file + ' ' + self.new_xdmf_file
			os.system(convert_command)

			self.config.set('Mesh', 'xdmf_file', self.new_xdmf_file)
			self.config.set('Mesh', 'gmsh_file', self.new_gmsh_file)
			self.config.set('OptimizationRoutine', 'rtol', '0.0')
			new_atol = self.shape_optimization_problem.solver.atol + self.shape_optimization_problem.solver.gradient_norm_initial*self.shape_optimization_problem.solver.rtol
			self.config.set('OptimizationRoutine', 'atol', str(new_atol))
			self.config.set('OptimizationRoutine', 'iteration_counter', str(self.shape_optimization_problem.solver.iteration))
			self.config.set('OptimizationRoutine', 'gradient_norm_initial', str(self.shape_optimization_problem.solver.gradient_norm_initial))

			config_path = self.config.get('Mesh', 'config_path')
			with open(config_path, 'w') as file:
				self.config.write(file)

			os.execv(sys.executable, ['python'] + sys.argv)
