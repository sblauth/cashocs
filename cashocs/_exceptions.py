"""
Created on 24/08/2020, 14.30

@author: blauths
"""



class cashocsException(Exception):
	"""Base class for exceptions raised by cashocs.

	"""

	pass



class NotConvergedError(cashocsException):
	"""This exception is raised when a solver does not converge.

	This includes any type of iterative method used to solve a problem,
	whether it is a linear or nonlinear system of equations, or an
	optimization problem.
	"""

	pass



class PETScKSPError(cashocsException):
	"""This exception is raised when the solution of a linear problem with PETSc fails.

	Also returns the PETSc error code and reason.
	"""

	def __init__(self, error_code, message='PETSc linear solver did not converge.'):
		self.message = message
		self.error_code = error_code

		if self.error_code == -2:
			self.error_reason = ' (ksp_diverged_null)'
		elif self.error_code == -3:
			self.error_reason = ' (ksp_diverged_its, reached maximum iterations)'
		elif self.error_code == -4:
			self.error_reason = ' (ksp_diverged_dtol, reached divergence tolerance)'
		elif self.error_code == -5:
			self.error_reason = ' (ksp_diverged_breakdown, krylov method breakdown)'
		elif self.error_code == -6:
			self.error_reason = ' (ksp_diverged_breakdown_bicg)'
		elif self.error_code == -7:
			self.error_reason = ' (ksp_diverged_nonsymmetric, need a symmetric operator / preconditioner)'
		elif self.error_code == -8:
			self.error_reason = ' (ksp_diverged_indefinite_pc, the preconditioner is indefinite, but needs to be positive definite)'
		elif self.error_code == -9:
			self.error_reason = ' (ksp_diverged_nanorinf)'
		elif self.error_code == -10:
			self.error_reason = ' (ksp_diverged_indefinite_mat, operator is indefinite, but needs to be positive definite)'
		elif self.error_code == -11:
			self.error_reason = ' (ksp_diverged_pc_failed, it was not possible to build / use the preconditioner)'
		else:
			self.error_reason = ' (unknown)'

	def __str__(self):
		return f'{self.message} KSPConvergedReason = {self.error_code} {self.error_reason}'



class InputError(cashocsException):
	"""This exception gets raised when the user input to a public API method is wrong or inconsistent.

	"""

	pass



class ConfigError(cashocsException):
	"""This exception gets raised when parameters in the config file are wrong.

	"""

	pass
