"""
Created on 14/08/2020, 08.57

@author: blauths
"""

from ._forms import Lagrangian



class OptimizationProblem:
	"""A class representing abstract optimization problems.

	This is subclassed to create both optimal control and
	shape optimization problems, which share a lot of common
	parameters.
	"""

	def __init__(self, state_forms, bcs_list, cost_functional_form, states, adjoints, config, initial_guess=None):
		"""Initializes the optimization problem

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
		config : configparser.ConfigParser
			the config file for the problem, generated via caospy.create_config(path_to_config)
		initial_guess : list[dolfin.function.function.Function], optional
			list of functions that act as initial guess for the state variables, should be valid input for fenics.assign.
			(defaults to None (which means a zero initial guess))
		"""

		### Overloading, so that we do not have to use lists for a single state and a single control
		### state_forms
		try:
			if type(state_forms) == list and len(state_forms) > 0:
				for i in range(len(state_forms)):
					if state_forms[i].__module__=='ufl.form' and type(state_forms[i]).__name__=='Form':
						pass
					else:
						raise SystemExit('state_forms have to be ufl forms')
				self.state_forms = state_forms
			elif state_forms.__module__ == 'ufl.form' and type(state_forms).__name__ == 'Form':
				self.state_forms = [state_forms]
			else:
				raise SystemExit('State forms have to be ufl forms')
		except:
			raise SystemExit('Type of state_forms is wrong.')
		self.state_dim = len(self.state_forms)

		### bcs_list
		try:
			if bcs_list == [] or bcs_list is None:
				self.bcs_list = []
				for i in range(self.state_dim):
					self.bcs_list.append([])
			elif type(bcs_list) == list and len(bcs_list) > 0:
				if type(bcs_list[0]) == list:
					for i in range(len(bcs_list)):
						if type(bcs_list[i]) == list:
							pass
						else:
							raise SystemExit('bcs_list has inconsistent types.')
					self.bcs_list = bcs_list

				elif bcs_list[0].__module__ == 'dolfin.fem.dirichletbc' and type(bcs_list[0]).__name__ == 'DirichletBC':
					for i in range(len(bcs_list)):
						if bcs_list[i].__module__=='dolfin.fem.dirichletbc' and type(bcs_list[i]).__name__=='DirichletBC':
							pass
						else:
							raise SystemExit('bcs_list has inconsistent types.')
					self.bcs_list = [bcs_list]
			elif bcs_list.__module__ == 'dolfin.fem.dirichletbc' and type(bcs_list).__name__ == 'DirichletBC':
				self.bcs_list = [[bcs_list]]
			else:
				raise SystemExit('Type of bcs_list is wrong.')
		except:
			raise SystemExit('Type of bcs_list is wrong.')

		### cost_functional_form
		try:
			if cost_functional_form.__module__ == 'ufl.form' and type(cost_functional_form).__name__ == 'Form':
				self.cost_functional_form = cost_functional_form
			else:
				raise SystemExit('cost_functional_form has to be a ufl form')
		except:
			raise SystemExit('Type of cost_functional_form is wrong.')

		### states
		try:
			if type(states) == list and len(states) > 0:
				for i in range(len(states)):
					if states[i].__module__ == 'dolfin.function.function' and type(states[i]).__name__ == 'Function':
						pass
					else:
						raise SystemExit('states have to be fenics Functions.')

				self.states = states

			elif states.__module__ == 'dolfin.function.function' and type(states).__name__ == 'Function':
				self.states = [states]
			else:
				raise SystemExit('Type of states is wrong.')
		except:
			raise SystemExit('Type of states is wrong.')

		### adjoints
		try:
			if type(adjoints) == list and len(adjoints) > 0:
				for i in range(len(adjoints)):
					if adjoints[i].__module__ == 'dolfin.function.function' and type(adjoints[i]).__name__ == 'Function':
						pass
					else:
						raise SystemExit('adjoints have to fenics Functions.')

				self.adjoints = adjoints

			elif adjoints.__module__ == 'dolfin.function.function' and type(adjoints).__name__ == 'Function':
				self.adjoints = [adjoints]
			else:
				raise SystemExit('Type of adjoints is wrong.')
		except:
			raise SystemExit('Type of adjoints is wrong.')

		### config
		if config.__module__ == 'configparser' and type(config).__name__ == 'ConfigParser':
			self.config = config
		else:
			raise SystemExit('config has to be of configparser.ConfigParser type')

		### initial guess
		if initial_guess is None:
			self.initial_guess = initial_guess
		else:
			try:
				if type(initial_guess) == list:
					self.initial_guess = initial_guess
				elif initial_guess.__module__ == 'dolfin.function.function' and type(initial_guess).__name__ == 'Function':
					self.initial_guess = [initial_guess]
			except:
				raise SystemExit('Initial guess has to be a list of functions')


		assert len(self.bcs_list) == self.state_dim, 'Length of states does not match'
		assert len(self.states) == self.state_dim, 'Length of states does not match'
		assert len(self.adjoints) == self.state_dim, 'Length of states does not match'
		if self.initial_guess is not None:
			assert len(self.initial_guess) == self.state_dim, 'Length of states does not match'

		self.state_problem = None
		self.adjoint_problem = None



	def compute_state_variables(self):
		"""Solves the state system

		This can be used for debugging purposes, to validate the
		solver and general behavior. Updates the user input for
		the state variables.

		Returns
		-------
		None
		"""

		self.state_problem.solve()



	def compute_adjoint_variables(self):
		"""Solves the adjoint system

		This can be used for debugging purposes, solver validation.
		Updates the user input for the adjoint variables. The solve
		to the corresponding state system needed to determine the
		adjoints is carried out automatically.

		Returns
		-------
		None
		"""

		self.adjoint_problem.solve()
