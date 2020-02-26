"""
Created on 26/02/2020, 11.13

@author: blauths
"""

from ..forms import Lagrangian, FormHandler
from ..pde_problems.state_problem import StateProblem
from ..pde_problems.adjoint_problem import AdjointProblem
from ..pde_problems.gradient_problem import GradientProblem
from ..optimization.cost_functional import CostFunctional
from ..pde_problems.hessian_problem import HessianProblem
from .methods.gradient_descent import GradientDescent
from .methods.l_bfgs import LBFGS
from .methods.CG import CG
from .methods.newton import Newton



class OptimizationProblem:
	def __init__(self, state_form, bcs, control_measure, cost_functional_form, state, control, adjoint, config):
		self.state_form = state_form
		self.bcs = bcs
		
		self.control_measure = control_measure
		
		self.cost_functional_form = cost_functional_form
		
		self.state = state
		self.control = control
		self.adjoint = adjoint
		
		self.config = config
		
		
		self.state_space = self.state.function_space()
		self.control_space = self.control.function_space()
		
		self.lagrangian = Lagrangian(self.state_form, self.cost_functional_form)
		
		self.form_handler = FormHandler(self.lagrangian, self.bcs, self.control_measure, self.state, self.control, self.adjoint, self.config)
		
		self.state_problem = StateProblem(self.form_handler)
		self.adjoint_problem = AdjointProblem(self.form_handler, self.state_problem)
		self.gradient_problem = GradientProblem(self.form_handler, self.state_problem, self.adjoint_problem)
		self.hessian_problem = HessianProblem(self.form_handler, self.gradient_problem)
		
		self.cost_functional = CostFunctional(self.form_handler, self.state_problem)
		
		self.gradient = self.gradient_problem.gradient
		
		
	def solve(self):
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
