"""
Created on 26/02/2020, 11.13

@author: blauths
"""

import fenics
from ..forms import Lagrangian, FormHandler
from ..pde_problems.state_problem import StateProblem
from ..pde_problems.adjoint_problem import AdjointProblem
from ..pde_problems.gradient_problem import GradientProblem
from ..optimization.cost_functional import ReducedCostFunctional
from ..pde_problems.hessian_problem import HessianProblem
from ..pde_problems.semi_smooth_hessian import SemiSmoothHessianProblem
from .methods.gradient_descent import GradientDescent
from .methods.l_bfgs import LBFGS
from .methods.CG import CG
from .methods.newton import Newton
from .methods.semi_smooth_newton import SemiSmoothNewton



class OptimizationProblem:
	
	def __init__(self, state_forms, bcs_list, control_measures, cost_functional_form, states, controls, adjoints, config, control_constraints=None, ramping_parameters=None):
		"""The implementation of the optimization problem, used to generate all other classes and functionality. Also used to solve the problem.
		
		Parameters
		----------
		state_forms : ufl.form.Form or List[ufl.form.Form]
			the weak form of the state equation (user implemented)
		bcs_list : List[dolfin.fem.dirichletbc.DirichletBC] or List[List[dolfin.fem.dirichletbc.DirichletBC]]
			the list of DirichletBC objects describing essential boundary conditions
		control_measures : ufl.measure.Measure or List[ufl.measure.Measure]
			the measure corresponding to the domain of the control
		cost_functional_form : ufl.form.Form
			the cost functional (user implemented)
		states : dolfin.function.function.Function or List[dolfin.function.function.Function]
			the state variable
		controls : dolfin.function.function.Function or List[dolfin.function.function.Function]
			the control variable
		adjoints : dolfin.function.function.Function or List[dolfin.function.function.Function]
			the adjoint variable
		config : configparser.ConfigParser
			the config file for the problem
		control_constraints : List[dolfin.function.function.Function] or List[float] or List[List]
			Box constraints posed on the control
		"""
		
		### Overloading, such that we do not have to use lists for single state single control
		if type(state_forms) == list:
			self.state_forms = state_forms
		else:
			self.state_forms = [state_forms]

		try:
			if type(bcs_list) == list and type(bcs_list[0]) == list:
				self.bcs_list = bcs_list
		except IndexError:
			self.bcs_list = [bcs_list]

		if type(control_measures) == list:
			self.control_measures = control_measures
		else:
			self.control_measures = [control_measures]
		
		self.cost_functional_form = cost_functional_form
		
		if type(states) == list:
			self.states = states
		else:
			self.states = [states]
		
		if type(controls) == list:
			self.controls = controls
		else:
			self.controls = [controls]
		
		if type(adjoints) == list:
			self.adjoints = adjoints
		else:
			self.adjoints = [adjoints]
		
		self.config = config

		if control_constraints is None:
			self.control_constraints = []
			for j in range(len(self.controls)):
				self.control_constraints.append([float('-inf'), float('inf')])
		else:
			if type(control_constraints) == list and type(control_constraints[0]) == list:
				self.control_constraints = control_constraints
			else:
				self.control_constraints = [control_constraints]

		self.ramping_parameters = ramping_parameters
		
		self.state_dim = len(self.state_forms)
		self.control_dim = len(self.controls)
		
		assert len(self.bcs_list) == self.state_dim, 'Length of states does not match'
		assert len(self.control_measures) == self.control_dim, 'Length of controls does not match'
		assert len(self.states) == self.state_dim, 'Length of states does not match'
		assert len(self.adjoints) == self.state_dim, 'Length of states does not match'
		assert len(self.control_constraints) == self.control_dim, 'Length of controls does not match'
		### end overloading
		
		self.state_spaces = [x.function_space() for x in self.states]
		self.control_spaces = [x.function_space() for x in self.controls]
		
		self.lagrangian = Lagrangian(self.state_forms, self.cost_functional_form)
		self.form_handler = FormHandler(self.lagrangian, self.bcs_list, self.control_measures, self.states, self.controls, self.adjoints, self.config, self.control_constraints)

		self.projected_difference = [fenics.Function(V) for V in self.control_spaces]

		self.state_problem = StateProblem(self.form_handler, self.ramping_parameters)
		self.adjoint_problem = AdjointProblem(self.form_handler, self.state_problem)
		self.gradient_problem = GradientProblem(self.form_handler, self.state_problem, self.adjoint_problem)
		
		if self.config.get('OptimizationRoutine', 'algorithm') == 'newton':
			self.hessian_problem = HessianProblem(self.form_handler, self.gradient_problem, self.control_constraints)
		if self.config.get('OptimizationRoutine', 'algorithm') == 'semi_smooth_newton':
			self.semi_smooth_hessian = SemiSmoothHessianProblem(self.form_handler, self.gradient_problem, self.control_constraints)
			
		self.reduced_cost_functional = ReducedCostFunctional(self.form_handler, self.state_problem)

		self.gradients = self.gradient_problem.gradients
		self.objective_value = 1.0



	def stationary_measure_squared(self):

		for j in range(self.control_dim):
			self.projected_difference[j].vector()[:] = self.controls[j].vector()[:] - self.gradients[j].vector()[:]

		self.form_handler.project(self.projected_difference)

		for j in range(self.control_dim):
			self.projected_difference[j].vector()[:] = self.controls[j].vector()[:] - self.projected_difference[j].vector()[:]

		return self.form_handler.scalar_product(self.projected_difference, self.projected_difference)


		
	def solve(self):
		"""Solves the optimization problem by the method specified in the config file. See adpack.optimization.methds for details on the implemented solution methods
		
		Returns
		-------
		
			Updates self.state, self.control and self.adjoint according to the optimization method. The user inputs for generating the OptimizationProblem class are actually manipulated there.

		"""
		
		self.algorithm = self.config.get('OptimizationRoutine', 'algorithm')
		
		if self.algorithm == 'gradient_descent':
			solver = GradientDescent(self)
		elif self.algorithm == 'lbfgs':
			solver = LBFGS(self)
		elif self.algorithm == 'cg':
			solver = CG(self)
		elif self.algorithm == 'newton':
			solver = Newton(self)
		elif self.algorithm == 'semi_smooth_newton':
			solver = SemiSmoothNewton(self)
		else:
			raise SystemExit('OptimizationRoutine.algorithm needs to be one of gradient_descent, lbfgs, cg or newton.')
		
		solver.run()
