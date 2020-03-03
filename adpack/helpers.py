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
