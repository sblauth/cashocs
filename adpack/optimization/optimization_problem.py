"""
Created on 26/02/2020, 11.13

@author: blauths
"""

from ..forms import Lagrangian, FormHandler
from ..pde_problems.state_problem import StateProblem
from ..pde_problems.adjoint_problem import AdjointProblem
from ..pde_problems.gradient_problem import GradientProblem
from ..optimization.cost_functional import ReducedCostFunctional
from ..pde_problems.hessian_problem import HessianProblem
from .methods.gradient_descent import GradientDescent
from .methods.l_bfgs import LBFGS
from .methods.CG import CG
from .methods.newton import Newton



class OptimizationProblem:
	
	def __init__(self, state_forms, bcs_list, control_measures, cost_functional_form, states, controls, adjoints, config):
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
		"""
		
		### Overloading, such that we do not have to use lists for single state single control
		if type(state_forms) == list:
			self.state_forms = state_forms
		else:
			self.state_forms = [state_forms]
		
		if type(bcs_list) == list and type(bcs_list[0]) == list:
			self.bcs_list = bcs_list
		else:
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
		
		self.state_dim = len(self.state_forms)
		self.control_dim = len(self.controls)
		
		assert len(self.bcs_list) == self.state_dim, 'Length of states does not match'
		assert len(self.control_measures) == self.control_dim, 'Length of controls does not match'
		assert len(self.states) == self.state_dim, 'Length of states does not match'
		assert len(self.adjoints) == self.state_dim, 'Length of states does not match'
		### end overloading
		
		self.state_spaces = [x.function_space() for x in self.states]
		self.control_spaces = [x.function_space() for x in self.controls]
		
		self.lagrangian = Lagrangian(self.state_forms, self.cost_functional_form)
		self.form_handler = FormHandler(self.lagrangian, self.bcs_list, self.control_measures, self.states, self.controls, self.adjoints, self.config)
		
		self.state_problem = StateProblem(self.form_handler)
		self.adjoint_problem = AdjointProblem(self.form_handler, self.state_problem)
		self.gradient_problem = GradientProblem(self.form_handler, self.state_problem, self.adjoint_problem)
		
		if self.config.get('OptimizationRoutine', 'algorithm') == 'newton':
			self.hessian_problem = HessianProblem(self.form_handler, self.gradient_problem)
			
		self.reduced_cost_functional = ReducedCostFunctional(self.form_handler, self.state_problem)

		self.gradients = self.gradient_problem.gradients
		
		
		
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
		else:
			raise SystemExit('OptimizationRoutine.algorithm needs to be one of gradient_descent, lbfgs, cg or newton.')
		
		solver.run()
