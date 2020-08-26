"""
Created on 25/08/2020, 08.28

@author: blauths
"""

import fenics
import numpy as np
from .._exceptions import NotConvergedError, ConfigError, CestrelException
from petsc4py import PETSc
from ..utils import _assemble_petsc_system, _solve_linear_problem, _setup_petsc_options



class BaseHessianProblem:
	"""Base class for derived Hessian problems.

	"""

	def __init__(self, form_handler, gradient_problem):
		"""Initializes self.

		Parameters
		----------
		form_handler : cestrel._forms.ControlFormHandler
			The FormHandler object for the optimization problem.
		gradient_problem : cestrel._pde_problems.GradientProblem
			The GradientProblem object (this is needed for the computation
			of the Hessian).
		"""

		self.form_handler = form_handler
		self.gradient_problem = gradient_problem

		self.config = self.form_handler.config
		self.gradients = self.gradient_problem.gradients

		self.inner_newton = self.config.get('OptimizationRoutine', 'inner_newton', fallback='cr')
		self.max_it_inner_newton = self.config.getint('OptimizationRoutine', 'max_it_inner_newton', fallback=50)
		# TODO: Add absolute tolerance, too
		self.inner_newton_tolerance = self.config.getfloat('OptimizationRoutine', 'inner_newton_tolerance', fallback=1e-15)

		self.test_directions = self.form_handler.test_directions
		self.residual = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.delta_control = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.state_dim = self.form_handler.state_dim
		self.control_dim = self.form_handler.control_dim

		self.temp1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.temp2 = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.p = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.p_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.p_pprev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.s = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.s_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.s_pprev = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.q = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.q_prev = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.hessian_actions = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.inactive_part = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.active_part = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.controls = self.form_handler.controls

		self.rtol = self.config.getfloat('StateEquation', 'picard_rtol', fallback=1e-10)
		self.atol = self.config.getfloat('StateEquation', 'picard_atol', fallback=1e-20)
		self.maxiter = self.config.getint('StateEquation', 'picard_iter', fallback=50)
		self.picard_verbose = self.config.getboolean('StateEquation', 'picard_verbose', fallback=False)

		self.no_sensitivity_solves = 0
		self.state_ksps = [PETSc.KSP().create() for i in range(self.form_handler.state_dim)]
		_setup_petsc_options(self.state_ksps, self.form_handler.state_ksp_options)
		self.adjoint_ksps = [PETSc.KSP().create() for i in range(self.form_handler.state_dim)]
		_setup_petsc_options(self.adjoint_ksps, self.form_handler.adjoint_ksp_options)



	def hessian_application(self, h, out):
		r"""Computes the application of the Hessian to some element

		This is needed in the truncated Newton method where we solve the system

		$$ J''(u) [\delta u] = - J'(u)
		$$

		via iterative methods (conjugate gradient or conjugate residual method)

		Parameters
		----------
		h : list[dolfin.function.function.Function]
			A function to which we want to apply the Hessian to.
		out : list[dolfin.function.function.Function]
			A list of functions into which the result is saved.

		Returns
		-------
		None
		"""

		for i in range(self.control_dim):
			self.test_directions[i].vector()[:] = h[i].vector()[:]

		self.states_prime = self.form_handler.states_prime
		self.adjoints_prime = self.form_handler.adjoints_prime
		self.bcs_list_ad = self.form_handler.bcs_list_ad

		if not self.form_handler.state_is_picard:

			for i in range(self.state_dim):
				A, b = _assemble_petsc_system(self.form_handler.sensitivity_eqs_lhs[i], self.form_handler.sensitivity_eqs_rhs[i], self.bcs_list_ad[i])
				_solve_linear_problem(self.state_ksps[i], A, b, self.states_prime[i].vector().vec())

			for i in range(self.state_dim):
				A, b = _assemble_petsc_system(self.form_handler.adjoint_sensitivity_eqs_lhs[-1 - i], self.form_handler.w_1[-1 - i], self.bcs_list_ad[-1 - i])
				_solve_linear_problem(self.adjoint_ksps[-1 - i], A, b, self.adjoints_prime[-1 - i].vector().vec())

		else:
			for i in range(self.maxiter + 1):
				res = 0.0
				for j in range(self.form_handler.state_dim):
					res_j = fenics.assemble(self.form_handler.sensitivity_eqs_picard[j])
					[bc.apply(res_j) for bc in self.form_handler.bcs_list_ad[j]]
					res += pow(res_j.norm('l2'), 2)

				if res==0:
					break

				res = np.sqrt(res)

				if i==0:
					res_0 = res

				if self.picard_verbose:
					print('Picard Sensitivity 1 Iteration ' + str(i) + ': ||res|| (abs): ' + format(res, '.3e') + '   ||res|| (rel): ' + format(res/res_0, '.3e'))

				if res/res_0 < self.rtol or res < self.atol:
					break

				if i==self.maxiter:
					raise NotConvergedError('Failed to solve the Picard iteration.')

				for j in range(self.form_handler.state_dim):
					A, b = _assemble_petsc_system(self.form_handler.sensitivity_eqs_lhs[j], self.form_handler.sensitivity_eqs_rhs[j], self.bcs_list_ad[j])
					_solve_linear_problem(self.state_ksps[j], A, b, self.states_prime[j].vector().vec())

			if self.picard_verbose:
				print('')

			for i in range(self.maxiter + 1):
				res = 0.0
				for j in range(self.form_handler.state_dim):
					res_j = fenics.assemble(self.form_handler.adjoint_sensitivity_eqs_picard[j])
					[bc.apply(res_j) for bc in self.form_handler.bcs_list_ad[j]]
					res += pow(res_j.norm('l2'), 2)

				if res==0:
					break

				res = np.sqrt(res)

				if i==0:
					res_0 = res

				if self.picard_verbose:
					print('Picard Sensitivity 2 Iteration ' + str(i) + ': ||res|| (abs): ' + format(res, '.3e') + '   ||res|| (rel): ' + format(res/res_0, '.3e'))

				if res/res_0 < self.rtol or res < self.atol:
					break

				if i==self.maxiter:
					raise NotConvergedError('Failed to solve the Picard iteration.')

				for j in range(self.form_handler.state_dim):
					A, b = _assemble_petsc_system(self.form_handler.adjoint_sensitivity_eqs_lhs[-1 - j], self.form_handler.w_1[-1 - j], self.bcs_list_ad[-1 - j])
					_solve_linear_problem(self.adjoint_ksps[-1 - j], A, b, self.adjoints_prime[-1 - j].vector().vec())

			if self.picard_verbose:
				print('')

		for i in range(self.control_dim):
			b = fenics.as_backend_type(fenics.assemble(self.form_handler.hessian_rhs[i])).vec()
			x = out[i].vector().vec()
			self.form_handler.ksps[i].solve(b, x)

			if self.form_handler.ksps[i].getConvergedReason() < 0:
				raise Exception('Krylov solver did not converge. Reason: ' + str(self.form_handler.ksps[i].getConvergedReason()))

		self.no_sensitivity_solves += 2



	def newton_solve(self, idx_active=None):

		self.gradient_problem.solve()
		self.form_handler.compute_active_sets()

		for j in range(self.control_dim):
			self.delta_control[j].vector()[:] = 0.0

		if self.inner_newton == 'cg':
			self.cg(idx_active)
		elif self.inner_newton == 'cr':
			self.cr(idx_active)
		else:
			raise ConfigError('Not a valid choice for OptimizationRoutine.inner_newton. Needs to be either cg or cr.')

		return self.delta_control



	def cg(self, idx_active=None):
		pass



	def cr(self, idx_active=None):
		pass





class HessianProblem(BaseHessianProblem):
	"""PDE Problem used to solve the (reduced) Hessian problem.

	"""

	def __init__(self, form_handler, gradient_problem):
		"""Initializes self.

		Parameters
		----------
		form_handler : cestrel._forms.ControlFormHandler
			The FormHandler object for the optimization problem.
		gradient_problem : cestrel._pde_problems.GradientProblem
			The GradientProblem object (this is needed for the computation
			of the Hessian).
		"""

		BaseHessianProblem.__init__(self, form_handler, gradient_problem)



	def reduced_hessian_application(self, h, out):

		for j in range(self.control_dim):
			out[j].vector()[:] = 0.0

		self.form_handler.restrict_to_inactive_set(h, self.inactive_part)
		self.hessian_application(self.inactive_part, self.hessian_actions)
		self.form_handler.restrict_to_inactive_set(self.hessian_actions, self.inactive_part)
		self.form_handler.restrict_to_active_set(h, self.active_part)

		for j in range(self.control_dim):
			out[j].vector()[:] = self.active_part[j].vector()[:] + self.inactive_part[j].vector()[:]



	def newton_solve(self, idx_active=None):

		if idx_active is not None:
			raise CestrelException('Must not pass idx_active to HessianProblem.')

		return BaseHessianProblem.newton_solve(self)



	def cg(self, idx_active=None):

		for j in range(self.control_dim):
			self.residual[j].vector()[:] = -self.gradients[j].vector()[:]
			self.p[j].vector()[:] = self.residual[j].vector()[:]

		self.rsold = self.form_handler.scalar_product(self.residual, self.residual)
		self.eps_0 = np.sqrt(self.rsold)

		for i in range(self.max_it_inner_newton):

			self.reduced_hessian_application(self.p, self.q)

			self.form_handler.restrict_to_active_set(self.p, self.temp1)
			sp_val1 = self.form_handler.scalar_product(self.temp1, self.temp1)

			self.form_handler.restrict_to_inactive_set(self.p, self.temp1)
			self.form_handler.restrict_to_inactive_set(self.q, self.temp2)
			sp_val2 = self.form_handler.scalar_product(self.temp1, self.temp2)
			sp_val = sp_val1 + sp_val2
			self.alpha = self.rsold / sp_val

			for j in range(self.control_dim):
				self.delta_control[j].vector()[:] += self.alpha * self.p[j].vector()[:]
				self.residual[j].vector()[:] -= self.alpha * self.q[j].vector()[:]

			self.rsnew = self.form_handler.scalar_product(self.residual, self.residual)
			self.eps = np.sqrt(self.rsnew)
			# print('Eps (CG): ' + str(self.eps / self.eps_0) + ' (rel)')
			if self.eps / self.eps_0 < self.inner_newton_tolerance:
				break

			self.beta = self.rsnew / self.rsold

			for j in range(self.control_dim):
				self.p[j].vector()[:] = self.residual[j].vector()[:] + self.beta * self.p[j].vector()[:]

			self.rsold = self.rsnew



	def cr(self, idx_active=None):

		for j in range(self.control_dim):
			self.residual[j].vector()[:] = -self.gradients[j].vector()[:]
			self.p[j].vector()[:] = self.residual[j].vector()[:]

		self.eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))

		self.reduced_hessian_application(self.residual, self.s)

		for j in range(self.control_dim):
			self.q[j].vector()[:] = self.s[j].vector()[:]

		self.form_handler.restrict_to_active_set(self.residual, self.temp1)
		self.form_handler.restrict_to_active_set(self.s, self.temp2)
		sp_val1 = self.form_handler.scalar_product(self.temp1, self.temp2)
		self.form_handler.restrict_to_inactive_set(self.residual, self.temp1)
		self.form_handler.restrict_to_inactive_set(self.s, self.temp2)
		sp_val2 = self.form_handler.scalar_product(self.temp1, self.temp2)

		self.rAr = sp_val1 + sp_val2

		for i in range(self.max_it_inner_newton):

			self.form_handler.restrict_to_active_set(self.q, self.temp1)
			self.form_handler.restrict_to_inactive_set(self.q, self.temp2)
			denom1 = self.form_handler.scalar_product(self.temp1, self.temp1)
			denom2 = self.form_handler.scalar_product(self.temp2, self.temp2)
			denominator = denom1 + denom2

			self.alpha = self.rAr / denominator

			for j in range(self.control_dim):
				self.delta_control[j].vector()[:] += self.alpha * self.p[j].vector()[:]
				self.residual[j].vector()[:] -= self.alpha * self.q[j].vector()[:]

			self.eps = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))
			# print('Eps (cr): ' + str(self.eps / self.eps_0) + ' (relative)')
			if self.eps / self.eps_0 < self.inner_newton_tolerance or i == self.max_it_inner_newton - 1:
				break

			self.reduced_hessian_application(self.residual, self.s)

			self.form_handler.restrict_to_active_set(self.residual, self.temp1)
			self.form_handler.restrict_to_active_set(self.s, self.temp2)
			sp_val1 = self.form_handler.scalar_product(self.temp1, self.temp2)
			self.form_handler.restrict_to_inactive_set(self.residual, self.temp1)
			self.form_handler.restrict_to_inactive_set(self.s, self.temp2)
			sp_val2 = self.form_handler.scalar_product(self.temp1, self.temp2)

			self.rAr_new = sp_val1 + sp_val2
			self.beta = self.rAr_new / self.rAr

			for j in range(self.control_dim):
				self.p[j].vector()[:] = self.residual[j].vector()[:] + self.beta * self.p[j].vector()[:]
				self.q[j].vector()[:] = self.s[j].vector()[:] + self.beta * self.q[j].vector()[:]

			self.rAr = self.rAr_new





class UnconstrainedHessianProblem(BaseHessianProblem):
	"""Hessian Problem without control constraints for the inner solver in PDAS.

	"""

	def __init__(self, form_handler, gradient_problem):
		"""Initializes self.

		Parameters
		----------
		form_handler : cestrel._forms.ControlFormHandler
			The FormHandler object for the optimization problem.
		gradient_problem : cestrel._pde_problems.GradientProblem
			The GradientProblem object (this is needed for the computation
			of the Hessian).
		"""

		BaseHessianProblem.__init__(self, form_handler, gradient_problem)

		self.reduced_gradient = [fenics.Function(self.form_handler.control_spaces[j]) for j in range(len(self.gradients))]
		self.temp = [fenics.Function(V) for V in self.form_handler.control_spaces]



	def reduced_hessian_application(self, h, out, idx_active):

		for j in range(self.control_dim):
			self.temp[j].vector()[:] = h[j].vector()[:]
			self.temp[j].vector()[idx_active[j]] = 0.0

		self.hessian_application(self.temp, out)

		for j in range(self.control_dim):
			out[j].vector()[idx_active[j]] = 0.0



	def newton_solve(self, idx_active=None):

		if idx_active is None:
			raise CestrelException('Need to pass idx_active to UnconstrainedHessianProblem.')

		self.gradient_problem.solve()

		for j in range(self.control_dim):
			self.reduced_gradient[j].vector()[:] = self.gradients[j].vector()[:]
			self.reduced_gradient[j].vector()[idx_active[j]] = 0.0

		return BaseHessianProblem.newton_solve(self, idx_active)



	def cg(self, idx_active=None):

		for j in range(self.control_dim):
			self.residual[j].vector()[:] = -self.reduced_gradient[j].vector()[:]
			self.p[j].vector()[:] = self.residual[j].vector()[:]

		self.rsold = self.form_handler.scalar_product(self.residual, self.residual)
		self.eps_0 = np.sqrt(self.rsold)

		for i in range(self.max_it_inner_newton):
			self.reduced_hessian_application(self.p, self.q, idx_active)

			self.alpha = self.rsold / self.form_handler.scalar_product(self.p, self.q)
			for j in range(self.control_dim):
				self.delta_control[j].vector()[:] += self.alpha * self.p[j].vector()[:]
				self.residual[j].vector()[:] -= self.alpha * self.q[j].vector()[:]

			self.rsnew = self.form_handler.scalar_product(self.residual, self.residual)
			self.eps = np.sqrt(self.rsnew)
			# print('Eps (CG): ' + str(self.eps / self.eps_0) + ' (rel)')
			if self.eps/self.eps_0 < self.inner_newton_tolerance:
				break

			self.beta = self.rsnew / self.rsold

			for j in range(self.control_dim):
				self.p[j].vector()[:] = self.residual[j].vector()[:] + self.beta * self.p[j].vector()[:]

			self.rsold = self.rsnew



	def cr(self, idx_active=None):

		for j in range(self.control_dim):
			self.residual[j].vector()[:] = -self.reduced_gradient[j].vector()[:]
			self.p[j].vector()[:] = self.residual[j].vector()[:]

		self.eps_0 = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))

		self.reduced_hessian_application(self.residual, self.s, idx_active)

		for j in range(self.control_dim):
			self.q[j].vector()[:] = self.s[j].vector()[:]

		self.rAr = self.form_handler.scalar_product(self.residual, self.s)

		for i in range(self.max_it_inner_newton):
			self.alpha = self.rAr / self.form_handler.scalar_product(self.q, self.q)

			for j in range(self.control_dim):
				self.delta_control[j].vector()[:] += self.alpha * self.p[j].vector()[:]
				self.residual[j].vector()[:] -= self.alpha * self.q[j].vector()[:]

			self.eps = np.sqrt(self.form_handler.scalar_product(self.residual, self.residual))
			# print('Eps (cr): ' + str(self.eps / self.eps_0) + ' (relative)')
			if self.eps/self.eps_0 < self.inner_newton_tolerance or i==self.max_it_inner_newton - 1:
				break

			self.reduced_hessian_application(self.residual, self.s, idx_active)

			self.rAr_new = self.form_handler.scalar_product(self.residual, self.s)
			self.beta = self.rAr_new / self.rAr

			for j in range(self.control_dim):
				self.p[j].vector()[:] = self.residual[j].vector()[:] + self.beta * self.p[j].vector()[:]
				self.q[j].vector()[:] = self.s[j].vector()[:] + self.beta * self.q[j].vector()[:]

			self.rAr = self.rAr_new





class SemiSmoothHessianProblem(BaseHessianProblem):
	"""Hessian problem for the semi smooth Newton method.

	"""

	def __init__(self, form_handler, gradient_problem):
		"""Initializes self.

		Parameters
		----------
		form_handler : cestrel._forms.ControlFormHandler
			The FormHandler object for the optimization problem.
		gradient_problem : cestrel._pde_problems.GradientProblem
			The GradientProblem object (this is needed for the computation
			of the Hessian).
		"""

		BaseHessianProblem.__init__(self, form_handler, gradient_problem)

		self.control_constraints = self.form_handler.control_constraints
		self.residual1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.residual2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.delta_mu = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.p1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.p2 = [fenics.Function(V) for V in self.form_handler.control_spaces]

		self.mu = [fenics.Function(V) for V in self.form_handler.control_spaces]
		for j in range(self.control_dim):
			self.mu[j].vector()[:] = -1.0
		self.temp = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.temp_storage = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.cc = 1e-4



	def double_scalar_product(self, a1, a2, b1, b2):
		"""Computes the scalar product for functions with two components.

		Parameters
		----------
		a1 : list[dolfin.function.function.Function]
			The first component of the first argument.
		a2 : list[dolfin.function.function.Function]
			The second component of the first argument.
		b1 : list[dolfin.function.function.Function]
			The first component of the second argument.
		b2 : list[dolfin.function.function.Function]
			The second component of the second argument.

		Returns
		-------
		float
			The value of the scalar product.
		"""

		s1 = self.form_handler.scalar_product(a1, b1)
		s2 = self.form_handler.scalar_product(a2, b2)

		return s1 + s2



	def compute_semi_smooth_active_sets(self):

		self.idx_active_lower = [(self.mu[j].vector()[:] + self.cc*(self.controls[j].vector()[:] - self.control_constraints[j][0].vector()[:]) < 0.0).nonzero()[0] for j in range(self.control_dim)]
		self.idx_active_upper = [(self.mu[j].vector()[:] + self.cc*(self.controls[j].vector()[:] - self.control_constraints[j][1].vector()[:]) > 0.0).nonzero()[0] for j in range(self.control_dim)]

		self.idx_active = [np.concatenate((self.idx_active_lower[j], self.idx_active_upper[j])) for j in range(self.control_dim)]
		[self.idx_active[j].sort() for j in range(self.control_dim)]

		self.idx_inactive = [np.setdiff1d(np.arange(self.form_handler.control_spaces[j].dim()), self.idx_active[j] ) for j in range(self.control_dim)]

		return None



	def restrict_to_active_set(self, a, b):
		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_active[j]] = a[j].vector()[self.idx_active[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def restrict_to_lower_active_set(self, a, b):
		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_active_lower[j]] = a[j].vector()[self.idx_active_lower[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def restrict_to_upper_active_set(self, a, b):
		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_active_upper[j]] = a[j].vector()[self.idx_active_upper[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def restrict_to_inactive_set(self, a, b):
		for j in range(self.control_dim):
			self.temp[j].vector()[:] = 0.0
			self.temp[j].vector()[self.idx_inactive[j]] = a[j].vector()[self.idx_inactive[j]]
			b[j].vector()[:] = self.temp[j].vector()[:]

		return b



	def newton_solve(self, idx_active=None):

		if idx_active is not None:
			raise CestrelException('Must not pass idx_active to SemiSmoothHessianProblem.')

		self.compute_semi_smooth_active_sets()

		for j in range(self.control_dim):
			self.delta_mu[j].vector()[:] = 0.0

		BaseHessianProblem.newton_solve(self)
		return self.delta_control, self.delta_mu



	def cg(self, idx_active=None):

		self.temp_storage1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.temp_storage2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.Ap1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.Ap2 = [fenics.Function(V) for V in self.form_handler.control_spaces]

		for j in range(self.control_dim):
			self.temp_storage1[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][0].vector()[:]
			self.temp_storage2[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][1].vector()[:]

		self.restrict_to_active_set(self.mu, self.temp_storage)
		self.restrict_to_lower_active_set(self.temp_storage1, self.temp_storage1)
		self.restrict_to_upper_active_set(self.temp_storage2, self.temp_storage2)

		for j in range(self.control_dim):
			self.residual1[j].vector()[:] = -self.gradients[j].vector()[:] - self.temp_storage[j].vector()[:]
			self.residual2[j].vector()[:] = - self.temp_storage1[j].vector()[:] - self.temp_storage2[j].vector()[:]
			self.p1[j].vector()[:] = self.residual1[j].vector()[:]
			self.p2[j].vector()[:] = self.residual2[j].vector()[:]

		self.res_norm_squared = self.double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2)
		self.eps_0 = np.sqrt(self.res_norm_squared)

		for i in range(self.max_it_inner_newton):
			self.hessian_application(self.p1, self.hessian_actions)
			self.restrict_to_active_set(self.p2, self.temp_storage1)
			self.restrict_to_active_set(self.p1, self.temp_storage2)

			for j in range(self.control_dim):
				self.Ap1[j].vector()[:] = self.hessian_actions[j].vector()[:] + self.temp_storage1[j].vector()[:]
				self.Ap2[j].vector()[:] = self.temp_storage2[j].vector()[:]

			self.eps = np.sqrt(self.res_norm_squared)
			print('Eps (CG): ' + str(self.eps / self.eps_0) + ' (rel)')
			if self.eps / self.eps_0 < self.inner_newton_tolerance:
				break

			self.alpha = self.res_norm_squared / self.double_scalar_product(self.p1, self.p2, self.Ap1, self.Ap2)
			for j in range(self.control_dim):
				self.delta_control[j].vector()[:] += self.alpha*self.p1[j].vector()[:]
				self.delta_mu[j].vector()[:] += self.alpha*self.p2[j].vector()[:]
				self.residual1[j].vector()[:] -= self.alpha*self.Ap1[j].vector()[:]
				self.residual2[j].vector()[:] -= self.alpha*self.Ap2[j].vector()[:]

			self.res_norm_squared = self.double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2)
			self.beta = self.res_norm_squared / pow(self.eps, 2)

			for j in range(self.control_dim):
				self.p1[j].vector()[:] = self.residual1[j].vector()[:] + self.beta*self.p1[j].vector()[:]
				self.p2[j].vector()[:] = self.residual2[j].vector()[:] + self.beta*self.p2[j].vector()[:]



	def cr(self, idx_active=None):

		self.temp_storage1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.temp_storage2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.Ar1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.Ar2 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.Ap1 = [fenics.Function(V) for V in self.form_handler.control_spaces]
		self.Ap2 = [fenics.Function(V) for V in self.form_handler.control_spaces]

		for j in range(self.control_dim):
			self.temp_storage1[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][0].vector()[:]
			self.temp_storage2[j].vector()[:] = self.controls[j].vector()[:] - self.control_constraints[j][1].vector()[:]

		self.restrict_to_active_set(self.mu, self.temp_storage)
		self.restrict_to_lower_active_set(self.temp_storage1, self.temp_storage1)
		self.restrict_to_upper_active_set(self.temp_storage2, self.temp_storage2)

		for j in range(self.control_dim):
			self.residual1[j].vector()[:] = -self.gradients[j].vector()[:] - self.temp_storage[j].vector()[:]
			self.residual2[j].vector()[:] = -self.temp_storage1[j].vector()[:] - self.temp_storage2[j].vector()[:]
			self.p1[j].vector()[:] = self.residual1[j].vector()[:]
			self.p2[j].vector()[:] = self.residual2[j].vector()[:]

		self.eps_0 = np.sqrt(self.double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2))

		self.hessian_application(self.residual1, self.hessian_actions)
		self.restrict_to_active_set(self.residual1, self.temp_storage1)
		self.restrict_to_active_set(self.residual2, self.temp_storage2)

		for j in range(self.control_dim):
			self.Ar1[j].vector()[:] = self.hessian_actions[j].vector()[:] + self.temp_storage2[j].vector()[:]
			self.Ar2[j].vector()[:] = self.temp_storage1[j].vector()[:]
			self.Ap1[j].vector()[:] = self.Ar1[j].vector()[:]
			self.Ap2[j].vector()[:] = self.Ar2[j].vector()[:]

		self.rAr = self.double_scalar_product(self.residual1, self.residual2, self.Ar1, self.Ar2)

		for i in range(self.max_it_inner_newton):
			# self.alpha = self.__double_scalar_product(self.residual1, self.residual2, self.Ap1, self.Ap2) / self.__double_scalar_product(self.Ap1, self.Ap2, self.Ap1, self.Ap2)
			self.alpha = self.rAr / self.double_scalar_product(self.Ap1, self.Ap2, self.Ap1, self.Ap2)

			for j in range(self.control_dim):
				self.delta_control[j].vector()[:] += self.alpha*self.p1[j].vector()[:]
				self.delta_mu[j].vector()[:] += self.alpha*self.p2[j].vector()[:]

				self.residual1[j].vector()[:] -= self.alpha*self.Ap1[j].vector()[:]
				self.residual2[j].vector()[:] -= self.alpha*self.Ap2[j].vector()[:]

			self.eps = np.sqrt(self.double_scalar_product(self.residual1, self.residual2, self.residual1, self.residual2))
			print('Eps (cr): ' + str(self.eps / self.eps_0) + ' (relative)')
			if self.eps / self.eps_0 < self.inner_newton_tolerance or i == self.max_it_inner_newton - 1:
				break

			self.hessian_application(self.residual1, self.hessian_actions)
			self.restrict_to_active_set(self.residual1, self.temp_storage1)
			self.restrict_to_active_set(self.residual2, self.temp_storage2)

			for j in range(self.control_dim):
				self.Ar1[j].vector()[:] = self.hessian_actions[j].vector()[:] + self.temp_storage2[j].vector()[:]
				self.Ar2[j].vector()[:] = self.temp_storage1[j].vector()[:]

			self.rAr_new = self.double_scalar_product(self.residual1, self.residual2, self.Ar1, self.Ar2)
			self.beta = self.rAr_new / self.rAr
			# self.beta = -self.__double_scalar_product(self.Ar1, self.Ar2, self.Ap1, self.Ap2) / self.__double_scalar_product(self.Ap1, self.Ap2, self.Ap1, self.Ap2)

			for j in range(self.control_dim):
				self.p1[j].vector()[:] = self.residual1[j].vector()[:] + self.beta*self.p1[j].vector()[:]
				self.p2[j].vector()[:] = self.residual2[j].vector()[:] + self.beta*self.p2[j].vector()[:]

				self.Ap1[j].vector()[:] = self.Ar1[j].vector()[:] + self.beta*self.Ap1[j].vector()[:]
				self.Ap2[j].vector()[:] = self.Ar2[j].vector()[:] + self.beta*self.Ap2[j].vector()[:]

			self.rAr = self.rAr_new
