"""
Created on 15/06/2020, 08.00

@author: blauths
"""

from ..optimization_problem import OptimizationProblem
from .._forms import Lagrangian, ShapeFormHandler
from .._pde_problems import StateProblem, AdjointProblem, ShapeGradientProblem
from .._shape_optimization import ReducedShapeCostFunctional
from .methods import LBFGS, CG, GradientDescent
from ..geometry import _MeshHandler



class ShapeOptimizationProblem(OptimizationProblem):
	"""A shape optimization problem

	This class is used to define a shape optimization problem, and also to solve
	it subsequently. For a detailed documentation, see the examples in the "demos"
	folder. For easier input, when consider single (state or control) variables,
	these do not have to be wrapped into a list.
	Note, that in the case of multiple variables these have to be grouped into
	ordered lists, where state_forms, bcs_list, states, adjoints have to have
	the same order (i.e. [state1, state2, state3,...] and [adjoint1, adjoint2,
	adjoint3, ...], where adjoint1 is the adjoint of state1 and so on.
	"""

	def __init__(self, state_forms, bcs_list, cost_functional_form, states,
				 adjoints, boundaries, config, initial_guess=None):
		"""Initializes the shape optimization problem

		This is used to generate all classes and functionalities. First ensures
		consistent input as the __init__ function is overloaded. Afterwards, the
		solution algorithm is initialized.

		Parameters
		----------
		state_forms : ufl.form.Form or list[ufl.form.Form]
			the weak form of the state equation (user implemented). Can be either
		bcs_list : list[dolfin.fem.dirichletbc.DirichletBC] or list[list[dolfin.fem.dirichletbc.DirichletBC]]
				   or dolfin.fem.dirichletbc.DirichletBC or None
			the list of DirichletBC objects describing Dirichlet (essential) boundary conditions.
			If this is None, then no Dirichlet boundary conditions are imposed.
		cost_functional_form : ufl.form.Form
			UFL form of the cost functional
		states : dolfin.function.function.Function or list[dolfin.function.function.Function]
			the state variable(s), can either be a single fenics Function, or a (ordered) list of these
		adjoints : dolfin.function.function.Function or list[dolfin.function.function.Function]
			the adjoint variable(s), can either be a single fenics Function, or a (ordered) list of these
		boundaries : dolfin.cpp.mesh.MeshFunctionSizet
			MeshFunction that indicates the boundary markers
		config : configparser.ConfigParser
			the config file for the problem, generated via adoptpy.create_config(path_to_config)
		initial_guess : list[dolfin.function.function.Function], optional
			list of functions that act as initial guess for the state variables, should be valid input for fenics.assign.
			(defaults to None (which means a zero initial guess))
		"""

		OptimizationProblem.__init__(self, state_forms, bcs_list, cost_functional_form, states, adjoints, config, initial_guess)

		### boundaries
		if boundaries.__module__ == 'dolfin.cpp.mesh' and type(boundaries).__name__ == 'MeshFunctionSizet':
			self.boundaries = boundaries
		else:
			raise SystemExit('Not a valid format for boundaries.')

		assert len(self.bcs_list) == self.state_dim, 'Length of states does not match'
		assert len(self.states) == self.state_dim, 'Length of states does not match'
		assert len(self.adjoints) == self.state_dim, 'Length of states does not match'
		if self.initial_guess is not None:
			assert len(self.initial_guess) == self.state_dim, 'Length of states does not match'

		self.lagrangian = Lagrangian(self.state_forms, self.cost_functional_form)
		self.shape_form_handler = ShapeFormHandler(self.lagrangian, self.bcs_list, self.states, self.adjoints, self.boundaries, self.config)
		self.mesh_handler = _MeshHandler(self)

		self.state_spaces = self.shape_form_handler.state_spaces
		self.adjoint_spaces = self.shape_form_handler.adjoint_spaces

		self.state_problem = StateProblem(self.shape_form_handler, self.initial_guess)
		self.adjoint_problem = AdjointProblem(self.shape_form_handler, self.state_problem)
		self.shape_gradient_problem = ShapeGradientProblem(self.shape_form_handler, self.state_problem, self.adjoint_problem)

		self.reduced_cost_functional = ReducedShapeCostFunctional(self.shape_form_handler, self.state_problem)

		self.gradient = self.shape_gradient_problem.gradient
		self.objective_value = 1.0



	def solve(self):
		"""Solves the optimization problem by the method specified in the config file.

		Returns
		-------
		None
		"""

		self.algorithm = self.config.get('OptimizationRoutine', 'algorithm')

		if self.algorithm in ['gradient_descent', 'gd']:
			self.solver = GradientDescent(self)
		elif self.algorithm in ['lbfgs', 'bfgs']:
			self.solver = LBFGS(self)
		elif self.algorithm in ['cg', 'conjugate_gradient']:
			self.solver = CG(self)
		else:
			raise SystemExit('OptimizationRoutine.algorithm needs to be one of gd, lbfgs, or cg.')

		self.solver.run()
		self.solver.finalize()



	def compute_shape_gradient(self):
		"""Solves the Riesz problem to determine the gradient(s)

		This can be used for debugging, or code validation.
		The necessary solutions of the state and adjoint systems
		are carried out automatically.

		Returns
		-------
		list[dolfin.function.function.Function]
			a list consisting of the (components) of the gradient
		"""

		self.shape_gradient_problem.solve()

		return self.gradient
