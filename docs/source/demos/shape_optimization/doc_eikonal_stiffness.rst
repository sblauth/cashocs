.. _demo_eikonal_stiffness:

Computing the Shape Stiffness via Distance to the Boundaries
============================================================

Problem Formulation
-------------------

We are solving the same problem as in :ref:`demo_shape_stokes`, but now use
a different approach for computing the stiffness of the shape gradient.
Recall, that the corresponding (regularized) shape optimization problem is given by

.. math::

    \min_\Omega J(u, \Omega) = &\int_{\Omega^\text{flow}} Du : Du\ \text{ d}x +
    \frac{\mu_\text{vol}}{2} \left( \int_\Omega 1 \text{ d}x - \text{vol}(\Omega_0) \right)^2 \\
    &+ \frac{\mu_\text{bary}}{2} \left\lvert \frac{1}{\text{vol}(\Omega)} \int_\Omega x \text{ d}x - \text{bary}(\Omega_0) \right\rvert^2 \\
    &\text{subject to } \quad \left\lbrace \quad
    \begin{alignedat}{2}
        - \Delta u + \nabla p &= 0 \quad &&\text{ in } \Omega, \\
        \text{div}(u) &= 0 \quad &&\text{ in } \Omega, \\
        u &= u^\text{in} \quad &&\text{ on } \Gamma^\text{in}, \\
        u &= 0 \quad &&\text{ on } \Gamma^\text{wall} \cup \Gamma^\text{obs}, \\
        \partial_n u - p n &= 0 \quad &&\text{ on } \Gamma^\text{out}.
    \end{alignedat}
    \right.

For a background on the stiffness of the shape gradient, we refer to :ref:`config_shape_shape_gradient`,
where it is defined as the parameter :math:`\mu` used in the computation of
the shape gradient. Note, that the distance computation is done via an eikonal equation,
hence the name of the demo.


Implementation
--------------

The complete python code can be found in the file :download:`demo_eikonal_stiffness.py </../../demos/documented/shape_optimization/eikonal_stiffness/demo_eikonal_stiffness.py>`,
and the corresponding config can be found in :download:`config.ini </../../demos/documented/shape_optimization/eikonal_stiffness/config.ini>`.


Changes in the config file
--------------------------

In order to compute the stiffness :math:`\mu` based on the distance to selected boundaries,
we only have to change the configuration file we are using, the python code
for solving the shape optimization problem with CASHOCS stays exactly as
it was in :ref:`demo_shape_stokes`.

To use the stiffness computation based on the distance to the boundary, we add the
following lines to the config file ::

    use_distance_mu = True
    dist_min = 0.05
    dist_max = 1.25
    mu_min = 5e2
    mu_max = 1.0
    smooth_mu = false
    boundaries_dist = [4]

The first line ::

    use_distance_mu = True

ensures that the stiffness will be computed based on the distance to the boundary.

The next four lines then specify the behavior of this computation. In particular,
we have the following behavior for :math:`\mu`

.. math::

    \mu = \begin{cases}
            \mu_\mathrm{min} \quad \text{ if } \delta \leq \delta_\mathrm{min},\\
            \mu_\mathrm{max} \quad \text{ if } \delta \geq \delta_\mathrm{max}
        \end{cases}

where :math:`\delta` denotes the distance to the boundary and :math:`\delta_\mathrm{min}`
and :math:`\delta_\mathrm{max}` correspond to ``dist_min`` and ``dist_max``,
respectively.

The values in-between are given by interpolation. Either a linear, continuous interpolation
is used, or a smooth :math:`C^1` interpolation given by a third order polynomial.
These can be selected with the option ::

    smooth_mu = false

where ``smooth_mu = True`` uses the third order polynomial, and ``smooth_mu = False`` uses
the linear function.

Finally, the line ::

    boundaries_dist = [4]

specifies, which boundaries are considered for the distance computation. These are
again specified using the boundary markers, as it was previously explained in
:ref:`config_shape_shape_gradient`.

The results should look like this

.. image:: /../../demos/documented/shape_optimization/shape_stokes/img_shape_stokes.png
