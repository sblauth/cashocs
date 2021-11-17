.. _demo_p_laplacian:

Shape Optimization with the p-Laplacian
=======================================

Problem Formulation
-------------------


In this demo, we take a look at yet another possibility to compute the shape gradient and to use this method for solving shape optimization problems. Here, we investigate the approach of `Müller, Kühl, Siebenborn, Deckelnick, Hinze, and Rung <https://doi.org/10.1007/s00158-021-03030-x>`_ and use the :math:`p`-Laplacian in order to compute the shape gradient. 
As a model problem, we consider the following one, as in :ref:`demo_shape_poisson`:

.. math::

    &\min_\Omega J(u, \Omega) = \int_\Omega u \text{ d}x \\
    &\text{subject to} \quad \left\lbrace \quad
    \begin{alignedat}{2}
    -\Delta u &= f \quad &&\text{ in } \Omega,\\
    u &= 0 \quad &&\text{ on } \Gamma.
    \end{alignedat} \right.


For the initial domain, we use the unit disc :math:`\Omega = \{ x \in \mathbb{R}^2 \,\mid\, \lvert\lvert x \rvert\rvert_2 < 1 \}`, and the right-hand side :math:`f` is given by

.. math:: f(x) = 2.5 \left( x_1 + 0.4 - x_2^2 \right)^2 + x_1^2 + x_2^2 - 1.


Implementation
--------------

The complete python code can be found in the file :download:`demo_p_laplacian.py </../../demos/documented/shape_optimization/p_laplacian/demo_p_laplacian.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/shape_optimization/p_laplacian/config.ini>`.


Source Code
***********

The python source code for this example is completely identical to the one in :ref:`demo_shape_poisson`, so we do not cover this here again. The only changes occur in 
the configuration file, which we cover below.

Configuration File
******************

All the relevant changes appear in the ShapeGradient Section of the config file, where we now add the following three lines ::

    use_p_laplacian = True
    p_laplacian_power = 10
    p_laplacian_stabilization = 0.0

Here, ``use_p_laplacian`` is a boolean flag which indicates that we want to override the default behavior and use the :math:`p` Laplacian to compute the shape gradient instead of linear elasticity. In particular, this means that we solve the following equation to determine the shape gradient :math:`\mathcal{G}` 

.. math::

    \begin{aligned}
        &\text{Find } \mathcal{G} \text{ such that } \\
        &\qquad \int_\Omega \mu \left( \nabla \mathcal{G} : \nabla \mathcal{G} \right)^{\frac{p-2}{2}} \nabla \mathcal{G} : \nabla \mathcal{V} + \delta \mathcal{G} \cdot \mathcal{V} \text{ d}x = dJ(\Omega)[\mathcal{V}] \\
        &\text{for all } \mathcal{V}.
    \end{aligned}

Here, :math:`dJ(\Omega)[\mathcal{V}]` is the shape derivative. The parameter :math:`p` is defined via the config file parameter ``p_laplacian_power``, and is 10 for this example. Finally, it is possible to use a stabilized formulation of the :math:`p`-Laplacian equation shown above, where the stabilization parameter is determined via the config line parameter ``p_laplacian_stabilization``, which should be small (e.g. in the order of ``1e-3``). Moreover, :math:`\mu` is the stiffness parameter, which can be specified via the config file parameters ``mu_def`` and ``mu_fixed`` and works as usually (cf. :ref:`demo_shape_poisson`:). Finally, we have added the possibility to use the damping parameter :math:`\delta`, which is specified via the config file parameter ``damping_factor``, also in the Section ShapeGradient.

.. note::

    Note, that the :math:`p`-Laplace methods are only meant to work with the gradient descent method. Other methods, such as BFGS or NCG methods, might be able to work on certain problems, but you might encounter strange behavior of the methods.

Finally, we show the result of the optimization, which looks similar to the one obtained in :ref:`demo_shape_poisson`:

.. image:: /../../demos/documented/shape_optimization/shape_poisson/img_shape_poisson.png

