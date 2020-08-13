"""Ultility and "helper" functions.

This module includes certain utility and helper functionalities
intended to make programming simpler. Mostly considers UFL forms
(summation, products, measures) and config files.

"""

import fenics
import configparser
import os


def summation(x):
	"""Sums up elements of a list.

	Parameters
	----------
	x : list[ufl.form.Form] or list[int] or list[float]
		The list of entries that shall be summed up

	Returns
	-------
	y : ufl.form.Form or int or float
		Sum of input (same type as entries of input)

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
	"""Multiplies the elements of a list

	Parameters
	----------
	x : list[ufl.core.expr.Expr] or list[int] or list[float]
		The list whose entries shall be multiplied

	Returns
	-------
	y : ufl.core.expr.Expr or int or float
		The result of the multiplication
	
	"""
	
	if len(x) == 0:
		y = fenics.Constant(1.0)
		print('Careful, empty list handed to multiplication')
	else:
		y = x[0]
		
		for item in x[1:]:
			y *= item
			
	return y



class _EmptyMeasure:
	"""Implements an empty measure (e.g. of a null set)

	This is used for automatic measure generation, e.g., if
	the fixed boundary is empty for a shape optimization problem,
	and is used to avoid case distinctions.

	Attributes
	----------
	measure : ufl.measure.Measure
		A UFL measure corresponding to a mesh
	"""

	def __init__(self, measure):
		"""
		This implements an empty measure, needed for more flexibility for UFL forms

		Parameters
		----------
		measure : ufl.measure.Measure
			The underlying measure, typically just the domain measure dx
		"""

		self.measure = measure


	def __rmul__(self, other):
		"""
		Generate 0 as result by multiplying with Constant(0)*measure
		"""

		return fenics.Constant(0)*other*self.measure



def generate_measure(idx, measure):
	"""Generates a measure based on mesh markers.

	Generates a MeasureSum or EmptyMeasure object corresponding to
	measure and the subdomains / boundaries specified in idx. This
	is a convenient shortcut to writing dx(1) + dx(2) + dx(3) + ...
	in case many measures are involved, and allows specification
	via config files.

	Parameters
	----------
	idx : list[int]
		list of indices for the boundary / volume markers
	measure : ufl.measure.Measure
		the corresponding ufl measure

	Returns
	-------
	out_measure
		The corresponding sum of the measures or an empty measure
	"""

	if len(idx) == 0:
		out_measure = _EmptyMeasure(measure)

	else:
		out_measure = measure(idx[0])

		for i in idx[1:]:
			out_measure += measure(i)

	return out_measure



def create_config(path):
	"""Generates a config object from a config file

	Creates the config from a .ini file via the
	configparser package, and also adds the path of the
	config file to the config. This is helpful for remeshing.

	Parameters
	----------
	path : str
		path to the config .ini file

	Returns
	-------
	configparser.ConfigParser
		the modified config file
	"""

	config = configparser.ConfigParser()
	config.read(path)
	config.set('Mesh', 'config_path', os.path.abspath(path))

	return config
