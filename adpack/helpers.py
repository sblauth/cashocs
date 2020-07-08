"""
Created on 02/03/2020, 13.58

@author: blauths
"""

import fenics



def summ(x):
	"""Computes the sum of a list in a fenics friendly fashion
	
	Parameters
	----------
	x : List
		The list of entries that shall be summed up

	Returns
	-------
	y
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
	x : List
		The list whose entries shall be multiplied

	Returns
	-------
	y
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
		This implements an empty measure, needed for more flexibility in programming

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
	Generates a MeasureSum or empty measure object corresponding to measure and the subdomains / boundaries specified in idx

	Parameters
	----------
	idx : list
	measure : ufl.measure.Measure

	Returns
	-------
	out_measure : ufl.measure.MeasureSum
		The outgoing measure, either a MeasureSum or an EmptyMeasure
	"""

	if len(idx) == 0:
		out_measure = EmptyMeasure(measure)

	else:
		out_measure = measure(idx[0])

		for i in idx[1:]:
			out_measure += measure(i)

	return out_measure
