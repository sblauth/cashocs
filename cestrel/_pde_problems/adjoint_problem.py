"""
Created on 24/02/2020, 09.24

@author: blauths
"""

import fenics
import numpy as np
from petsc4py import PETSc
from ..utils import _assemble_petsc_system, _setup_petsc_options



class AdjointProblem:
	"""The adjoint problem

	This class implements the adjoint problem as well as its solver.
	"""
	
	def __init__(self, form_handler, state_problem, temp_dict=None):
		"""Initializes the AdjointProblem
		
		Parameters
		----------
		form_handler : cestrel._forms.ControlFormHandler or cestrel._forms.ShapeFormHandler
			the FormHandler object for the optimization problem
		state_problem : cestrel._pde_problems.StateProblem
			the StateProblem object used to get the point where we linearize the problem
		temp_dict : dict
			A dictionary used for reinitializations when remeshing is performed.
		"""
		
		self.form_handler = form_handler
		self.state_problem = state_problem
		self.temp_dict = temp_dict
		
		self.config = self.form_handler.config
		self.adjoints = self.form_handler.adjoints
		self.bcs_list_ad = self.form_handler.bcs_list_ad

		self.rtol = self.config.getfloat('StateEquation', 'picard_rtol', fallback=1e-10)
		self.atol = self.config.getfloat('StateEquation', 'picard_atol', fallback=1e-12)
		self.maxiter = self.config.getint('StateEquation', 'picard_iter', fallback=50)
		self.picard_verbose = self.config.getboolean('StateEquation', 'picard_verbose', fallback=False)

		self.ksps = [PETSc.KSP().create() for i in range(self.form_handler.state_dim)]
		_setup_petsc_options(self.ksps, self.form_handler.adjoint_ksp_options)

		try:
			self.number_of_solves = self.temp_dict['output_dict'].get('adjoint_solves', 0)
		except TypeError:
			self.number_of_solves = 0
		self.has_solution = False
	
	
	
	def solve(self):
		"""Solves the adjoint system
		
		Returns
		-------
		adjoints : list[dolfin.function.function.Function]
			list of adjoint variables
		"""
		
		self.state_problem.solve()

		if not self.has_solution:
			if not self.form_handler.state_is_picard:
				for i in range(self.form_handler.state_dim):
					A, b = _assemble_petsc_system(self.form_handler.adjoint_eq_lhs[-1 - i], self.form_handler.adjoint_eq_rhs[-1 - i], self.bcs_list_ad[-1 - i])

					self.ksps[-1 - i].setOperators(A)
					self.ksps[-1 - i].solve(b, self.adjoints[-1 - i].vector().vec())

			else:
				for i in range(self.maxiter + 1):
					res = 0.0
					for j in range(self.form_handler.state_dim):
						res_j = fenics.assemble(self.form_handler.adjoint_picard_forms[j])
						[bc.apply(res_j) for bc in self.form_handler.bcs_list_ad[j]]
						res += pow(res_j.norm('l2'), 2)

					if res==0:
						break

					res = np.sqrt(res)

					if i==0:
						res_0 = res

					if self.picard_verbose:
						print('Iteration ' + str(i) + ': ||res|| (abs): ' + format(res, '.3e') + '   ||res|| (rel): ' + format(res/res_0, '.3e'))

					if res/res_0 < self.rtol or res < self.atol:
						break

					if i==self.maxiter:
						raise Exception('Failed to solve the Picard Iteration')

					for j in range(self.form_handler.state_dim):
						A, b = _assemble_petsc_system(self.form_handler.adjoint_eq_lhs[-1 - j], self.form_handler.adjoint_eq_rhs[-1 - j], self.bcs_list_ad[-1 - j])
						self.ksps[-1 - j].setOperators(A)
						self.ksps[-1 - j].solve(b, self.adjoints[-1 - j].vector().vec())
						# fenics.PETScOptions.set('mat_mumps_icntl_24', 1)
						# fenics.solve(self.form_handler.adjoint_eq_lhs[-1 - j]==self.form_handler.adjoint_eq_rhs[-1 - j], self.adjoints[-1 - j], self.bcs_list_ad[-1 - j], solver_parameters={'linear_solver' : 'mumps'})


			if self.picard_verbose:
				print('')
			self.has_solution = True
			self.number_of_solves += 1
		
		return self.adjoints
