"""
Created on 24/02/2020, 09.32

@author: blauths
"""

import fenics
import numpy as np



class OptimizationAlgorithm:
	
	def __init__(self, optimization_problem):
		"""Parent class for the optimization methods implemented in adpack.optimization.methods
		
		Parameters
		----------
		optimization_problem : adpack.optimization.optimization_problem.OptimizationProblem
			the OptimizationProblem class as defined through the user
		"""
		
		self.optimization_problem = optimization_problem
		self.form_handler = self.optimization_problem.form_handler
		self.state_problem = self.optimization_problem.state_problem
		self.config = self.state_problem.config
		self.adjoint_problem = self.optimization_problem.adjoint_problem


	
	def run(self):
		pass
