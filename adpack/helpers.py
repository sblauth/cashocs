"""
Created on 02/03/2020, 13.58

@author: blauths
"""

import fenics
import configparser
import os



def summ(x):
	"""Computes the sum of a list in a fenics friendly fashion
	
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



def prodd(x):
	"""Computes the product of a list in a fenics friendly fashion
	
	Parameters
	----------
	x : list[ufl.coefficient.Coefficient] or list[int] or list[float]
		The list whose entries shall be multiplied

	Returns
	-------
	y : ufl.coefficient.Coefficient or int or float
		The result of the multiplication
	
	"""
	
	if len(x) == 0:
		y = fenics.Constant(1.0)
		print('Careful, empty list handed to prodd')
	else:
		y = x[0]
		
		for item in x[1:]:
			y *= item
			
	return y



class EmptyMeasure:
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
		return fenics.Constant(0)*other*self.measure



def generate_measure(idx, measure):
	"""
	Generates a MeasureSum or EmptyMeasure object corresponding to measure and the subdomains / boundaries specified in idx

	Parameters
	----------
	idx : list[int]
		list of indices for the boundary / volume markers

	measure : ufl.measure.Measure
		the corresponding ufl measure

	Returns
	-------
	out_measure : ufl.measure.MeasureSum or adpack.helpers.EmptyMeasure
		The corresponding sum of the measures
	"""

	if len(idx) == 0:
		out_measure = EmptyMeasure(measure)

	else:
		out_measure = measure(idx[0])

		for i in idx[1:]:
			out_measure += measure(i)

	return out_measure



def create_config(path):
	"""Generates a configparser object and adds the config's path to it

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
