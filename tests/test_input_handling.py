"""
Created on 27/07/2021, 10.40

@author: blauths
"""

import os

import pytest
from fenics import *

import cashocs
from cashocs._exceptions import InputError



dir_path = os.path.dirname(os.path.realpath(__file__))
config = cashocs.load_config(dir_path + "/config_ocp.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
V = FunctionSpace(mesh, "CG", 1)

y = Function(V)
p = Function(V)
u = Function(V)

F = inner(grad(y), grad(p)) * dx - u * p * dx
bcs = cashocs.create_dirichlet_bcs(V, Constant(0), boundaries, [1, 2, 3, 4])

y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1, domain=mesh)
alpha = 1e-6
J = Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx


def test_inputs_ocp():
    with pytest.raises(InputError) as e_info:
        ocp = cashocs.OptimalControlProblem(1.0, bcs, J, y, u, p, config)
    assert "state_forms" in str(e_info.value)

    with pytest.raises(InputError) as e_info:
        ocp = cashocs.OptimalControlProblem(F, 1.0, J, y, u, p, config)
    assert "bcs_list" in str(e_info.value)

    with pytest.raises(InputError) as e_info:
        ocp = cashocs.OptimalControlProblem(F, bcs, 1.0, y, u, p, config)
    assert "cost_functional_form" in str(e_info.value)

    with pytest.raises(InputError) as e_info:
        ocp = cashocs.OptimalControlProblem(F, bcs, J, 1.0, u, p, config)
    assert "states" in str(e_info.value)

    with pytest.raises(InputError) as e_info:
        ocp = cashocs.OptimalControlProblem(F, bcs, J, y, 1.0, p, config)
    assert "controls" in str(e_info.value)

    with pytest.raises(InputError) as e_info:
        ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, 1.0, config)
    assert "adjoints" in str(e_info.value)

    with pytest.raises(InputError) as e_info:
        ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, 1.0)
    assert "config" in str(e_info.value)

    # with pytest.raises(InputError):
    #     ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config)
