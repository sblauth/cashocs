"""
Created on 1/30/19, 2:27 PM

@author: blauths
"""

import fenics
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
