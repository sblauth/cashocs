#!/usr/bin/env python

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

"""Mesh conversion from GMSH .msh to .xdmf.

"""

import argparse
import sys

import meshio



def _generate_parser():
	parser = argparse.ArgumentParser(description='Convert GMSH to XDMF.')
	parser.add_argument('infile', type=str, help='GMSH file that shall be converted, has to end in .msh')
	parser.add_argument('outfile', type=str, help='XDMF file into which the mesh shall be converted, has to end in .xdmf')
	
	return parser



def convert(argv=None):
	parser = _generate_parser()
	
	args = argv or parser.parse_args(argv)

	inputfile = args.infile
	outputfile = args.outfile
	# Check that the inputfile has .msh file format
	if not (inputfile[-4:] == '.msh'):
		print('Error: Cannot use the current file format as input.')
		sys.exit(2)

	# Check that the outputfile has .xdmf format
	if outputfile[-5:] == '.xdmf':
		oformat = '.xdmf'
		ostring = outputfile[:-5]
	else:
		print('Error: Cannot use the current file format as output.')
		sys.exit(2)

	mesh_collection = meshio.read(inputfile)

	points = mesh_collection.points
	cells_dict = mesh_collection.cells_dict
	cell_data_dict = mesh_collection.cell_data_dict

	# Check, whether we have a 2D or 3D mesh:
	keyvals = cells_dict.keys()
	if 'tetra' in keyvals:
		meshdim = 3
	elif 'triangle' in keyvals:
		meshdim = 2
	else:
		print('Error: This is not a valid input mesh.')
		sys.exit(2)

	if meshdim == 2:
		points = points[:, :2]
		xdmf_mesh = meshio.Mesh(points=points, cells={'triangle' : cells_dict['triangle']})
		meshio.write(ostring + '.xdmf', xdmf_mesh)

		if 'gmsh:physical' in cell_data_dict.keys():
			if 'triangle' in cell_data_dict['gmsh:physical'].keys():
				subdomains = meshio.Mesh(points=points, cells={'triangle': cells_dict['triangle']},
										 cell_data={'subdomains': [cell_data_dict['gmsh:physical']['triangle']]})
				meshio.write(ostring + '_subdomains.xdmf', subdomains)

			if 'line' in cell_data_dict['gmsh:physical'].keys():
				xdmf_boundaries = meshio.Mesh(points=points, cells={'line' : cells_dict['line']},
											  cell_data={'boundaries' : [cell_data_dict['gmsh:physical']['line']]})
				meshio.write(ostring + '_boundaries.xdmf', xdmf_boundaries)

	elif meshdim == 3:
		xdmf_mesh = meshio.Mesh(points=points, cells={'tetra' : cells_dict['tetra']})
		meshio.write(ostring + '.xdmf', xdmf_mesh)

		if 'gmsh:physical' in cell_data_dict.keys():
			if 'tetra' in cell_data_dict['gmsh:physical'].keys():
				subdomains = meshio.Mesh(points=points, cells={'tetra': cells_dict['tetra']},
										 cell_data={'subdomains': [cell_data_dict['gmsh:physical']['tetra']]})
				meshio.write(ostring + '_subdomains.xdmf', subdomains)

			if 'triangle' in cell_data_dict['gmsh:physical'].keys():
				xdmf_boundaries = meshio.Mesh(points=points, cells={'triangle' : cells_dict['triangle']},
											  cell_data={'boundaries' : [cell_data_dict['gmsh:physical']['triangle']]})
				meshio.write(ostring + '_boundaries.xdmf', xdmf_boundaries)



if __name__ == "__main__":
	parser = _generate_parser()
	convert(parser.parse_args())
