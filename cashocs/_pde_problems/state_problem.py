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

"""Abstract implementation of a state equation.

"""

import fenics
import numpy as np
from petsc4py import PETSc

from .._exceptions import NotConvergedError
from ..nonlinear_solvers import damped_newton_solve
from ..utils import _assemble_petsc_system, _setup_petsc_options, _solve_linear_problem



class StateProblem:
	"""The state system.

	"""

	def __init__(self, form_handler, initial_guess, temp_dict=None):
		"""Initializes the state system.

		Parameters
		----------
		form_handler : cashocs._forms.ControlFormHandler or cashocs._forms.ShapeFormHandler
			The FormHandler of the optimization problem.
		initial_guess : list[dolfin.function.function.Function]
			An initial guess for the state variables, used to initialize them in each iteration.
		temp_dict : dict
			A dict used for reinitialization when remeshing is performed.
		"""

		self.form_handler = form_handler
		self.initial_guess = initial_guess
		self.temp_dict = temp_dict

		self.config = self.form_handler.config
		self.bcs_list = self.form_handler.bcs_list
		self.states = self.form_handler.states

		self.rtol = self.config.getfloat('StateSystem', 'picard_rtol', fallback=1e-10)
		self.atol = self.config.getfloat('StateSystem', 'picard_atol', fallback=1e-20)
		self.maxiter = self.config.getint('StateSystem', 'picard_iter', fallback=50)
		self.picard_verbose = self.config.getboolean('StateSystem', 'picard_verbose', fallback=False)
		self.newton_rtol = self.config.getfloat('StateSystem', 'newton_rtol', fallback=1e-11)
		self.newton_atol = self.config.getfloat('StateSystem', 'newton_atol', fallback=1e-13)
		self.newton_damped = self.config.getboolean('StateSystem', 'newton_damped', fallback=True)
		self.newton_verbose = self.config.getboolean('StateSystem', 'newton_verbose', fallback=False)
		self.newton_iter = self.config.getint('StateSystem', 'newton_iter', fallback=50)

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
		"""Solves the state system.

		Returns
		-------
		states : list[dolfin.function.function.Function]
			The solution of the state system.
		"""

		if not self.has_solution:
			if self.initial_guess is not None:
				for j in range(self.form_handler.state_dim):
					fenics.assign(self.states[j], self.initial_guess[j])

			if not self.form_handler.state_is_picard or self.form_handler.state_dim == 1:
				if self.form_handler.state_is_linear:
					for i in range(self.form_handler.state_dim):
						A, b = _assemble_petsc_system(self.form_handler.state_eq_forms_lhs[i], self.form_handler.state_eq_forms_rhs[i], self.bcs_list[i])
						_solve_linear_problem(self.ksps[i], A, b, self.states[i].vector().vec(), self.form_handler.state_ksp_options[i])
						self.states[i].vector().apply('')

				else:
					for i in range(self.form_handler.state_dim):
						if self.initial_guess is not None:
							fenics.assign(self.states[i], self.initial_guess[i])

						self.states[i] = damped_newton_solve(self.form_handler.state_eq_forms[i], self.states[i], self.bcs_list[i],
															 rtol=self.newton_rtol, atol=self.newton_atol, max_iter=self.newton_iter,
															 damped=self.newton_damped, verbose=self.newton_verbose, ksp=self.ksps[i],
															 ksp_options=self.form_handler.state_ksp_options[i])

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
						raise NotConvergedError('Picard iteration for the state system')

					for j in range(self.form_handler.state_dim):
						if self.initial_guess is not None:
							fenics.assign(self.states[j], self.initial_guess[j])

						# adapt tolerances so that a solution is possible
						if not self.form_handler.state_is_linear:
							self.ksps[j].setTolerances(rtol=np.minimum(0.9*res, 0.9)/100, atol=self.newton_atols[j]/100)

							self.states[j] = damped_newton_solve(self.form_handler.state_eq_forms[j], self.states[j], self.bcs_list[j],
																 rtol=np.minimum(0.9*res, 0.9), atol=self.newton_atols[j], max_iter=self.newton_iter,
																 damped=self.newton_damped, verbose=self.newton_verbose, ksp=self.ksps[j],
																 ksp_options=self.form_handler.state_ksp_options[j])
						else:
							A, b = _assemble_petsc_system(self.form_handler.state_eq_forms_lhs[j], self.form_handler.state_eq_forms_rhs[j], self.bcs_list[j])
							_solve_linear_problem(self.ksps[j], A, b, self.states[j].vector().vec(), self.form_handler.state_ksp_options[j])
							self.states[j].vector().apply('')

			if self.picard_verbose and self.form_handler.state_is_picard:
				print('')
			self.has_solution = True
			self.number_of_solves += 1

		return self.states
