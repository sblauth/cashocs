"""
Created on 16/09/2020, 15.37

@author: blauths
"""

from fenics import *

_cpp_code_mesh_quality = """
			#include <pybind11/pybind11.h>
			#include <pybind11/eigen.h>
			namespace py = pybind11;

			#include <dolfin/mesh/Mesh.h>
			#include <dolfin/mesh/MeshFunction.h>
			#include <dolfin/mesh/Cell.h>

			using namespace dolfin;

			dolfin::MeshFunction<double>
			quality(std::shared_ptr<const Mesh> mesh)
			{
			  MeshFunction<double> cf(mesh, mesh->topology().dim(), 0.0);

			  for (CellIterator cell(*mesh); !cell.end(); ++cell)
			  {
				cf[*cell] = 1.0;
			  }
			  return cf;
			}

			PYBIND11_MODULE(SIGNATURE, m)
			{
			  m.def("quality", &quality);
			}

		"""

obj = compile_cpp_code(_cpp_code_mesh_quality)
mesh = UnitSquareMesh(10,10)
mf = obj.quality(mesh)
