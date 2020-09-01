# Copyright (C) 2020 Sebastian Blauth
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

"""
Created on 01/09/2020, 14.18

@author: blauths
"""

import pytest
import os
import fenics
import cashocs
import numpy as np
import filecmp



c_mesh, _, _, _, _, _ = cashocs.regular_mesh(5)
u_mesh = fenics.UnitSquareMesh(5,5)


def test_mesh_import():
	os.system('cashocs-convert ./mesh/mesh.msh ./mesh/mesh.xdmf')
	mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh('./mesh/mesh.xdmf')
	
	gmsh_coords = np.array([[0,0], [1,0], [1,1], [0,1], [0.499999999998694, 0], [1, 0.499999999998694], [0.5000000000020591, 1],
							[0, 0.5000000000020591], [0.2500000000010297, 0.7500000000010296], [0.3749999970924328, 0.3750000029075671],
							[0.7187499979760099, 0.2812500030636815], [0.6542968741702071, 0.6542968818888233]])
	
	assert abs(fenics.assemble(1*dx) - 1) < 1e-14
	assert abs(fenics.assemble(1*ds) - 4) < 1e-14
	
	assert abs(fenics.assemble(1*ds(1)) - 1) < 1e-14
	assert abs(fenics.assemble(1*ds(2)) - 1) < 1e-14
	assert abs(fenics.assemble(1*ds(3)) - 1) < 1e-14
	assert abs(fenics.assemble(1*ds(4)) - 1) < 1e-14
	
	assert np.allclose(mesh.coordinates(), gmsh_coords)
	
	os.system('rm ./mesh/mesh.xdmf')
	os.system('rm ./mesh/mesh.h5')
	os.system('rm ./mesh/mesh_subdomains.xdmf')
	os.system('rm ./mesh/mesh_subdomains.h5')
	os.system('rm ./mesh/mesh_boundaries.xdmf')
	os.system('rm ./mesh/mesh_boundaries.h5')



def test_regular_mesh():
	lens = np.random.uniform(0.5, 2, 2)
	r_mesh, _, _, _, _, _ = cashocs.regular_mesh(2, lens[0], lens[1])
	
	max_vals = np.random.uniform(0.5, 1, 3)
	min_vals = np.random.uniform(-1, -0.5, 3)
	
	s_mesh, _, _, _, _, _ = cashocs.regular_box_mesh(2, min_vals[0], min_vals[1], min_vals[2], max_vals[0], max_vals[1], max_vals[2])
	
	assert np.allclose(c_mesh.coordinates(), u_mesh.coordinates())
	
	assert np.alltrue((np.max(r_mesh.coordinates(), axis=0) - lens) < 1e-14)
	assert np.alltrue((np.min(r_mesh.coordinates(), axis=0) - np.array([0,0])) < 1e-14)
	
	assert np.alltrue(abs(np.max(s_mesh.coordinates(), axis=0) - max_vals) < 1e-14)
	assert np.alltrue(abs(np.min(s_mesh.coordinates(), axis=0) - min_vals) < 1e-14)



def test_mesh_quality_2D():
	mesh, _, _, _, _, _ = cashocs.regular_mesh(2)
	
	opt_angle = 60/360*2*np.pi
	alpha_1 = 90/360*2*np.pi
	alpha_2 = 45/360*2*np.pi
	
	q_1 = 1 - np.maximum((alpha_1 - opt_angle) / (np.pi - opt_angle), (opt_angle - alpha_1) / (opt_angle))
	q_2 = 1 - np.maximum((alpha_2 - opt_angle) / (np.pi - opt_angle), (opt_angle - alpha_2) / (opt_angle))
	q = np.minimum(q_1, q_2)
	
	min_max_angle = cashocs.MeshQuality.min_maximum_angle(mesh)
	min_radius_ratios = cashocs.MeshQuality.min_radius_ratios(mesh)
	average_radius_ratios = cashocs.MeshQuality.avg_radius_ratios(mesh)
	min_condition = cashocs.MeshQuality.min_condition_number(mesh)
	average_condition = cashocs.MeshQuality.avg_condition_number(mesh)
	
	assert abs(min_max_angle - cashocs.MeshQuality.avg_maximum_angle(mesh)) < 1e-14
	assert abs(min_max_angle - q) < 1e-14
	assert abs(min_max_angle - cashocs.MeshQuality.avg_skewness(mesh)) < 1e-14
	assert abs(min_max_angle - cashocs.MeshQuality.min_skewness(mesh)) < 1e-14
	
	assert abs(min_radius_ratios - average_radius_ratios) < 1e-14
	assert abs(min_radius_ratios - np.min(fenics.MeshQuality.radius_ratio_min_max(mesh))) < 1e-14
	
	assert abs(min_condition - average_condition) < 1e-14
	assert abs(min_condition - 0.4714045207910318) < 1e-14



def test_mesh_quality_3D():
	mesh, _, _, _, _, _ = cashocs.regular_mesh(2, 1.0, 1.0, 1.0)
	opt_angle = np.arccos(1/3)
	dh_min_max = fenics.MeshQuality.dihedral_angles_min_max(mesh)
	alpha_min = dh_min_max[0]
	alpha_max = dh_min_max[1]
	
	q_1 = 1 - np.maximum((alpha_max - opt_angle) / (np.pi - opt_angle), (opt_angle - alpha_max) / (opt_angle))
	q_2 = 1 - np.maximum((alpha_min - opt_angle) / (np.pi - opt_angle), (opt_angle - alpha_min) / (opt_angle))
	q = np.minimum(q_1, q_2)
	
	r_1 = 1 - np.maximum((alpha_max - opt_angle) / (np.pi - opt_angle), 0.0)
	r_2 = 1 - np.maximum((alpha_min - opt_angle) / (np.pi - opt_angle), 0.0)
	r = np.minimum(r_1, r_2)
	
	min_max_angle = cashocs.MeshQuality.min_maximum_angle(mesh)
	min_radius_ratios = cashocs.MeshQuality.min_radius_ratios(mesh)
	min_skewness = cashocs.MeshQuality.min_skewness(mesh)
	average_radius_ratios = cashocs.MeshQuality.avg_radius_ratios(mesh)
	min_condition = cashocs.MeshQuality.min_condition_number(mesh)
	average_condition = cashocs.MeshQuality.avg_condition_number(mesh)
	
	assert abs(min_max_angle - cashocs.MeshQuality.avg_maximum_angle(mesh)) < 1e-14
	assert abs(min_max_angle - r) < 1e-14
	
	assert abs(min_skewness - cashocs.MeshQuality.avg_skewness(mesh)) < 1e-14
	assert abs(min_skewness - q) < 1e-14
	
	assert abs(min_radius_ratios - average_radius_ratios) < 1e-14
	assert abs(min_radius_ratios - np.min(fenics.MeshQuality.radius_ratio_min_max(mesh))) < 1e-14
	
	assert abs(min_condition - average_condition) < 1e-14
	assert abs(min_condition - 0.3162277660168379) < 1e-14



def test_write_mesh():
	os.system('cashocs-convert ./mesh/mesh.msh ./mesh/mesh.xdmf')
	mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh('./mesh/mesh.xdmf')
	
	cashocs.utils.write_out_mesh(mesh, './mesh/mesh.msh', './mesh/test.msh')
	
	os.system('cashocs-convert ./mesh/test.msh ./mesh/test.xdmf')
	test, _, _, _, _, _ = cashocs.import_mesh('./mesh/test.xdmf')
	
	assert np.allclose(test.coordinates()[:, :], mesh.coordinates()[:, :])
	
	os.system('rm ./mesh/test.msh')
	os.system('rm ./mesh/test.xdmf')
	os.system('rm ./mesh/test.h5')
	os.system('rm ./mesh/test_subdomains.xdmf')
	os.system('rm ./mesh/test_subdomains.h5')
	os.system('rm ./mesh/test_boundaries.xdmf')
	os.system('rm ./mesh/test_boundaries.h5')
	
	os.system('rm ./mesh/mesh.xdmf')
	os.system('rm ./mesh/mesh.h5')
	os.system('rm ./mesh/mesh_subdomains.xdmf')
	os.system('rm ./mesh/mesh_subdomains.h5')
	os.system('rm ./mesh/mesh_boundaries.xdmf')
	os.system('rm ./mesh/mesh_boundaries.h5')



def test_create_measure():
	mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(5)
	meas = cashocs.utils.generate_measure([1,2,3], ds)
	test = ds(1) + ds(2) + ds(3)
	
	assert abs(fenics.assemble(1*meas) - 3) < 1e-14
	for i in range(3):
		assert meas._measures[i] == test._measures[i]
