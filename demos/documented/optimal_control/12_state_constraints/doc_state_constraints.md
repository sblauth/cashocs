## Demo 12 : State Constraints

In this demo we investigate how state constraints can be handled in cashocs. Thanks to
the high level interface for solving (control-constrained) optimal control problems,
the state constrained case can be treated (approximately) using a Moreau-Yosida regularization, which we show in the following. As model problem, we consider the
following one

$$\min\; J(y,u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2 \text{d}x + \frac{\alpha}{2} \int_{\Omega} u^2 \text{d}x \\
\text{ subject to } \quad \left\lbrace \quad
\begin{alignedat}{2}
-\Delta y &= u \quad &&\text{ in } \Omega,\\
y &= 0 \quad &&\text{ on } \Gamma, \\
y &\leq \bar{y} \quad &&\text{ in } \Omega,
\end{alignedat} \right.
$$

see, e.g., [Hinze et al., Optimization with PDE constraints](https://doi.org/10.1007/978-1-4020-8839-1)).

Instead of solving this problem directly, the Moreau-Yosida regularization instead solves
a sequence of problems without state constraints which are of the form

$$\min J_\gamma(y, u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2 \text{d}x + \frac{\alpha}{2} \int_{\Omega} u^2 \text{d}x + \frac{1}{2\gamma} \int_\Omega \lvert \max\left( 0, \hat{\mu} + \gamma (y - \bar{y}) \right) \rvert^2 \text{d}x
$$

for \( \gamma \to +\infty \). We employ a simple homotopy method, and solve the problem for one value of \( \gamma \), and then use this solution as initial guess for the next
higher value of \( \gamma \). As initial guess we use the solution of the unconstrained
problem. For a detailed discussion of the Moreau-Yosida regularization, we refer the
reader to, e.g., [Hinze et al., Optimization with PDE constraints](https://doi.org/10.1007/978-1-4020-8839-1)).

**Initial guess**

The python code starts as in [Demo 01](#demo-01-basics):

    from fenics import *
    import cashocs
    import numpy as np
    from ufl import Max



    config = cashocs.create_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
    V = FunctionSpace(mesh, 'CG', 1)

    y = Function(V)
    p = Function(V)
    u = Function(V)

    e = inner(grad(y), grad(p))*dx - u*p*dx
    bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

    y_d = Expression('sin(2*pi*x[0]*x[1])', degree=1)
    alpha = 1e-3
    J_init = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*dx
    ocp_init = cashocs.OptimalControlProblem(e, bcs, J_init, y, u, p, config)
    ocp_init.solve()

In fact, the code can be reused, as we first solve the unconstrained problem to get
the initial guess for the control. Remember that cashocs automatically updates the
user input, so after the solve command has returned, the solution is already stored in `u`.

**Moreau-Yosida regularized problems**

For the homotopy method with the Moreau-Yosida regularization, we first define the upper
bound for the state `\bar{y}` and select a sequence of values for \( \gamma \) via

    y_bar = 1e-1
    gammas = [pow(10, i) for i in np.arange(1, 9, 3)]

Solving the regularized problems is then as simple as writing

    for gamma in gammas:

    	J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*u*u*dx + Constant(1/(2*gamma))*pow(Max(0, Constant(gamma)*(y - y_bar)), 2)*dx

    	ocp_gamma = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config)
    	ocp_gamma.solve()

Here, we use a for loop, define the new cost functional (with the new value of \( \gamma \) ), set up the optimal control problem and solve it, as usual.

> Note, that we could have also defined `y_bar` as a FEniCS Function or Expression, and
> the method would have worked exactly the same, the corresponding object just has to
> be a valid input for an UFL form.

**Validation of the method**

Finally, we perform a post processing to see whether the state constraint is (approximately) satisfied. Therefore, we compute the maximum value of `y`, and compute the relative error between this and `y_bar`

    y_max = np.max(y.vector()[:])
    error = abs(y_max - y_bar) / abs(y_bar) * 100
    print('Maximum value of y: ' + str(y_max))
    print('Relative error between y_max and y_bar: ' + str(error) + ' %')

As the error is about 0.01 %, we observe that the regularization indeed works as expected, and of course this tolerance is sufficiently low for all applications.

The visualization of the solution looks as follows

![](./img/optimal_control/12_state_constraints.png)

The complete code can be found under demos/documented/optimal_control/12_state_constraints/demo_state_constraints.py
