"""
Created on 24/02/2020, 09.19

@author: blauths
"""

import fenics
import numpy as np
from ..nonlinear_solvers import damped_newton_solve
from petsc4py import PETSc
from ..utils import _assemble_petsc_system, _setup_petsc_options



class StateProblem:
	"""The state system

	"""

	def __init__(self, form_handler, initial_guess, temp_dict=None):
		"""Initialize the state system
		
		Parameters
		----------
		form_handler : cestrel._forms.ControlFormHandler or cestrel._forms.ShapeFormHandler
			the FormHandler of the optimization problem
		initial_guess : list[dolfin.function.function.Function]
			an initial guess for the state variables, used to initialize them in each iteration
		temp_dict : dict
			A dict used for reinitialization when remeshing is performed.
		"""
		
		self.form_handler = form_handler
		self.initial_guess = initial_guess
		self.temp_dict = temp_dict

		self.config = self.form_handler.config
		self.bcs_list = self.form_handler.bcs_list
		self.states = self.form_handler.states

		self.rtol = self.config.getfloat('StateEquation', 'picard_rtol', fallback=1e-10)
		self.atol = self.config.getfloat('StateEquation', 'picard_atol', fallback=1e-20)
		self.maxiter = self.config.getint('StateEquation', 'picard_iter', fallback=50)
		self.picard_verbose = self.config.getboolean('StateEquation', 'picard_verbose', fallback=False)
		self.newton_rtol = self.config.getfloat('StateEquation', 'inner_newton_rtol', fallback=1e-11)
		self.newton_atol = self.config.getfloat('StateEquation', 'inner_newton_atol', fallback=1e-13)
		self.newton_damped = self.config.getboolean('StateEquation', 'newton_damped', fallback=True)
		self.newton_verbose = self.config.getboolean('StateEquation', 'newton_verbose', fallback=False)
		self.newton_iter = self.config.getint('StateEquation', 'newton_iter', fallback=50)

		self.newton_atols = [1 for i in range(self.form_handler.state_dim)]

		self.ksps = [PETSc.KSP().create() for i in range(self.form_handler.state_dim)]
		_setup_petsc_options(self.ksps, self.form_handler.state_ksp_options)

		# adapt the tolerances so that the Newton system can be solved sucessfully
		if not self.form_handler.state_is_linear:
			for ksp in self.ksps:
				ksp.setTolerances(rtol=self.newton_rtol/100, atol=self.newton_atol/100)

		try:
			self.number_of_solves = self.temp_dict['output_dict'].get('state_solves', 0)
		except TypeError:
			self.number_of_solves = 0
		self.has_solution = False
	
	
	
	def solve(self):
		"""Solves the state system
		
		Returns
		-------
		states : list[dolfin.function.function.Function]
			the solution of the state system
		"""

		if not self.has_solution:
			if self.initial_guess is not None:
				for j in range(self.form_handler.state_dim):
					fenics.assign(self.states[j], self.initial_guess[j])

			if not self.form_handler.state_is_picard:
				if self.form_handler.state_is_linear:
					for i in range(self.form_handler.state_dim):
						A, b = _assemble_petsc_system(self.form_handler.state_eq_forms_lhs[i], self.form_handler.state_eq_forms_rhs[i], self.bcs_list[i])

						self.ksps[i].setOperators(A)
						self.ksps[i].solve(b, self.states[i].vector().vec())

				else:
					for i in range(self.form_handler.state_dim):
						if self.initial_guess is not None:
							fenics.assign(self.states[i], self.initial_guess[i])

						self.states[i] = damped_newton_solve(self.form_handler.state_eq_forms[i], self.states[i], self.bcs_list[i],
															 rtol=self.newton_rtol, atol=self.newton_atol, max_iter=self.newton_iter,
															 damped=self.newton_damped, verbose=self.newton_verbose, ksp=self.ksps[i])

			else:
				for i in range(self.maxiter + 1):
					res = 0.0
					for j in range(self.form_handler.state_dim):
						res_j = fenics.assemble(self.form_handler.state_picard_forms[j])

						[bc.apply(res_j) for bc  in self.form_handler.bcs_list_ad[j]]

						if self.number_of_solves==0 and i==0:
							self.newton_atols[j] = res_j.norm('l2') * self.newton_atol
							if res_j.norm('l2') == 0.0:
								self.newton_atols[j] = self.newton_atol

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
						if self.initial_guess is not None:
							fenics.assign(self.states[j], self.initial_guess[j])

						# adapt tolerances so that a solution is possible
						if not self.form_handler.state_is_linear:
							self.ksps[j].setTolerances(rtol=np.minimum(0.9*res, 0.9)/100, atol=self.newton_atols[j]/100)

						self.states[j] = damped_newton_solve(self.form_handler.state_eq_forms[j], self.states[j], self.bcs_list[j],
															 rtol=np.minimum(0.9*res, 0.9), atol=self.newton_atols[j], max_iter=self.newton_iter,
															 damped=self.newton_damped, verbose=self.newton_verbose, ksp=self.ksps[j])

			if self.picard_verbose:
				print('')
			self.has_solution = True
			self.number_of_solves += 1

		return self.states
