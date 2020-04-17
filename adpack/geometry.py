"""
Created on 1/30/19, 2:27 PM

@author: blauths
"""

import fenics
import numpy as np
import time



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
