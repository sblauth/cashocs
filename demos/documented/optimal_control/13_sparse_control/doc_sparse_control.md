## Demo 13 : Sparse Control

In this demo, we investigate a possibility for obtaining sparse optimal controls.
To do so, we use a sparsity promoting \( L^1 \) regularization. Hence, our model problem
for this demo is given by

$$\min\; J(y,u) = \frac{1}{2} \int_{\Omega} \left( y - y_d \right)^2 \text{d}x + \frac{\alpha}{2} \int_{\Omega} \lvert u \rvert \text{d}x \\
\text{ subject to } \quad \left\lbrace \quad
\begin{alignedat}{2}
-\Delta y &= u \quad &&\text{ in } \Omega,\\
y &= 0 \quad &&\text{ on } \Gamma.
\end{alignedat} \right.
$$

This is basically the same problem as in [Demo 01](#demo-01-basics), but the regularization is now not the \( L^2\) norm squared, but just the \( L^1 \) norm.


**Implementation**

The implementation of this problem is completely analogous to the one of [Demo 01](#demo-01-basics), the only difference is the definition of the cost functional. Hence, the entire code reads

from fenics import *
import cashocs



    config = cashocs.create_config('config.ini')
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(25)
    V = FunctionSpace(mesh, 'CG', 1)

    y = Function(V)
    p = Function(V)
    u = Function(V)

    e = inner(grad(y), grad(p))*dx - u*p*dx
    bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

    y_d = Expression('sin(2*pi*x[0])*sin(2*pi*x[1])', degree=1)
    alpha = 1e-4
    J = Constant(0.5)*(y - y_d)*(y - y_d)*dx + Constant(0.5*alpha)*abs(u)*dx

    ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config)
    ocp.solve()

Note, that for the regularization term we now do not use `Constant(0.5*alpha)*u*u*dx`,
which corresponds to the \( L^2 \) norm squared, but rather

    Constant(0.5*alpha)*abs(u)*dx

which corresponds to the \( L^1 \) norm. Other than that, the code is identical. To
verify that this yields in fact a sparse control, one can use the command

    plot(u)

which shows a plot of the computed optimal control. It should look like this
![](../demos/documented/optimal_control/13_sparse_control/sparse_control.png)

Note, that the oscillations in between the peaks are just numerical noise.
