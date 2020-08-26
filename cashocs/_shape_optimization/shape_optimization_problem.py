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
from .._exceptions import InputError, ConfigError
from ..utils import _optimization_algorithm_configuration
import sys
import tempfile
import os
import json
import warnings



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
				 adjoints, boundaries, config, initial_guess=None,
				 ksp_options=None, adjoint_ksp_options=None):
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
			the config file for the problem, generated via cashocs.create_config(path_to_config)
		initial_guess : list[dolfin.function.function.Function], optional
			A list of functions that act as initial guess for the state variables, should be valid input for fenics.assign.
			(defaults to None (which means a zero initial guess))
		ksp_options : list[list[str]] or list[list[list[str]]] or None, optional
			A list of strings corresponding to command line options for PETSc,
			used to solve the state systems. If this is None, then the direct solver
			mumps is used (default is None).
		adjoint_ksp_options : list[list[str]] or list[list[list[str]]] or None
			A list of strings corresponding to command line options for PETSc,
			used to solve the adjoint systems. If this is None, then the same options
			as for the state systems are used (default is None).
		"""

		OptimizationProblem.__init__(self, state_forms, bcs_list, cost_functional_form, states, adjoints, config, initial_guess, ksp_options, adjoint_ksp_options)

		### Initialize the remeshing behavior, and a temp file
		self.do_remesh = config.getboolean('Mesh', 'remesh', fallback=False)
		self.temp_dict = None
		if self.do_remesh:

			assert os.path.isfile(os.path.realpath(sys.argv[0])), 'Not a valid configuration. The script has to be the first "command line argument".'

			try:
				if __IPYTHON__:
					warnings.warn('You are running a shape optimization problem with remeshing from ipython. Rather run this using the python command')
			except NameError:
				pass

			try:
				assert self.states[0].function_space().mesh()._cashocs_generator == 'config', 'Can only handle the config file mesh import for remeshing'
			except:
				raise InputError('For remeshing, the mesh has to be created via import mesh, with a config as input.')

			if not ('_cashocs_remesh_flag' in sys.argv):
				self.directory = os.path.dirname(os.path.realpath(sys.argv[0]))
				self.__clean_previous_temp_files()
				self.temp_dir = tempfile.mkdtemp(prefix='._cashocs_remesh_temp', dir=self.directory)
				self.__change_except_hook()
				self.temp_dict = {'temp_dir' : self.temp_dir, 'gmsh_file' : self.config.get('Mesh', 'gmsh_file'),
								  'geo_file' : self.config.get('Mesh', 'geo_file'),
								  'OptimizationRoutine' : {'iteration_counter' : 0, 'gradient_norm_initial' : 0.0},
								  'output_dict' : {}}

			else:
				self.temp_dir = sys.argv[-1]
				self.__change_except_hook()
				with open(self.temp_dir + '/temp_dict.json', 'r') as file:
					self.temp_dict = json.load(file)

		### boundaries
		if boundaries.__module__ == 'dolfin.cpp.mesh' and type(boundaries).__name__ == 'MeshFunctionSizet':
			self.boundaries = boundaries
		else:
			raise InputError('Not a valid format for boundaries.')

		assert len(self.bcs_list) == self.state_dim, 'Length of states does not match'
		assert len(self.states) == self.state_dim, 'Length of states does not match'
		assert len(self.adjoints) == self.state_dim, 'Length of states does not match'
		if self.initial_guess is not None:
			assert len(self.initial_guess) == self.state_dim, 'Length of states does not match'

		self.lagrangian = Lagrangian(self.state_forms, self.cost_functional_form)
		self.shape_form_handler = ShapeFormHandler(self.lagrangian, self.bcs_list, self.states, self.adjoints,
												   self.boundaries, self.config, self.ksp_options, self.adjoint_ksp_options)
		self.mesh_handler = _MeshHandler(self)

		self.state_spaces = self.shape_form_handler.state_spaces
		self.adjoint_spaces = self.shape_form_handler.adjoint_spaces

		self.state_problem = StateProblem(self.shape_form_handler, self.initial_guess, self.temp_dict)
		self.adjoint_problem = AdjointProblem(self.shape_form_handler, self.state_problem, self.temp_dict)
		self.shape_gradient_problem = ShapeGradientProblem(self.shape_form_handler, self.state_problem, self.adjoint_problem)

		self.reduced_cost_functional = ReducedShapeCostFunctional(self.shape_form_handler, self.state_problem)

		self.gradient = self.shape_gradient_problem.gradient
		self.objective_value = 1.0



	def solve(self, algorithm=None, rtol=None, atol=None, max_iter=None):
		r"""Solves the optimization problem by the method specified in the config file.

		Parameters
		----------
		algorithm : str or None, optional
			Selects the optimization algorithm. Valid choices are
			'gradient_descent' ('gd'), 'conjugate_gradient' ('cg'),
			or 'lbfgs' ('bfgs'). This overwrites the value specified
			in the config file. If this is None, then the value in the
			config file is used. Default is None.
		rtol : float or None, optional
			The relative tolerance used for the termination criterion.
			Overwrites the value specified in the config file. If this
			is None, the value from the config file is taken. Default
			is None.
		atol : float or None, optional
			The absolute tolerance used for the termination criterion.
			Overwrites the value specified in the config file. If this
			is None, the value from the config file is taken. Default
			is None.
		max_iter : int or None, optional
			The maximum number of iterations the optimization algorithm
			can carry out before it is terminated. Overwrites the value
			specified in the config file. If this is None, the value from
			the config file is taken. Default is None.

		Returns
		-------
		None

		Notes
		-----
		If either `rtol` or `atol` are specified as arguments to the solve
		call, the termination criterion changes to:

		  - a purely relative one (if only `rtol` is specified), i.e.,
		$$ || \nabla J(u_k) || \leq \texttt{rtol} || \nabla J(u_0) ||.
		$$
		  - a purely absolute one (if only `atol` is specified), i.e.,
		$$ || \nabla J(u_K) || \leq \texttt{atol}.
		$$
		  - a combined one if both `rtol` and `atol` are specified, i.e.,
		$$ || \nabla J(u_k) || \leq \texttt{atol} + \texttt{rtol} || \nabla J(u_0) ||
		$$
		"""

		self.algorithm = _optimization_algorithm_configuration(self.config, algorithm)

		if (rtol is not None) and (atol is None):
			self.config.set('OptimizationRoutine', 'rtol', str(rtol))
			self.config.set('OptimizationRoutine', 'atol', str(0.0))
		elif (atol is not None) and (rtol is None):
			self.config.set('OptimizationRoutine', 'rtol', str(0.0))
			self.config.set('OptimizationRoutine', 'atol', str(atol))
		elif (atol is not None) and (rtol is not None):
			self.config.set('OptimizationRoutine', 'rtol', str(rtol))
			self.config.set('OptimizationRoutine', 'atol', str(atol))

		if max_iter is not None:
			self.config.set('OptimizationRoutine', 'maximum_iterations', str(max_iter))

		if self.algorithm == 'gradient_descent':
			self.solver = GradientDescent(self)
		elif self.algorithm == 'lbfgs':
			self.solver = LBFGS(self)
		elif self.algorithm == 'conjugate_gradient':
			self.solver = CG(self)
		else:
			raise ConfigError('Not a valid choice for OptimizationRoutine.algorithm. Needs to be one of `gradient_descent` (`gd`), `lbfgs` (`bfgs`), or `conjugate_gradient` (`cg`).')

		self.solver.run()
		self.solver.finalize()



	def __change_except_hook(self):
		"""Ensures that temp files are deleted when an exception occurs.

		This modifies the sys.excepthook command so that it also deletes temp files
		(only needed for remeshing)

		Returns
		-------
		None
		"""

		def custom_except_hook(exctype, value, traceback):
			print('DEBUG: Caught the exception, deleting temp files')
			os.system('rm -r ' + self.temp_dir)
			sys.__excepthook__(exctype, value, traceback)

		sys.excepthook = custom_except_hook



	def __clean_previous_temp_files(self):

		for file in os.listdir(self.directory):
			if file.startswith('._cashocs_remesh_temp'):
				os.system('rm -r ' + file)



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
