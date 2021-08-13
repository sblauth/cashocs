"""
Created on 13/08/2021, 09.06

@author: blauths
"""

import os
import cashocs
import cashocs.space_mapping.optimal_control as ocsm
import numpy as np

from fenics import *


dir_path = os.path.dirname(os.path.realpath(__file__))


def test_parameter_extraction_optimal_control_single():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
    V = FunctionSpace(mesh, "CG", 1)

    y = Function(V)
    p = Function(V)
    u = Function(V)

    F = inner(grad(y), grad(p)) * dx - u * p * dx
    bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

    y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1, domain=mesh)
    alpha = 1e-6
    J = Constant(0.5) * (y - y_d) * (y - y_d) * dx + Constant(0.5 * alpha) * u * u * dx

    ocp = cashocs.OptimalControlProblem(F, bcs, J, y, u, p, config=config)
    ocp.solve()
    control_ocp = u.vector()[:]

    u.vector()[:] = 0.0
    coarse_model = ocsm.CoarseModel(F, bcs, J, y, u, p, config=config)
    coarse_model.optimize()
    control_coarse_model = u.vector()[:]

    assert (
        np.max(np.abs(control_ocp - control_coarse_model)) / np.max(np.abs(control_ocp))
        < 1e-14
    )

    u_pe = Function(V)
    y_pe = Function(V)
    J_pe = (
        Constant(0.5) * (y_pe - y_d) * (y_pe - y_d) * dx
        + Constant(0.5 * alpha) * u_pe * u_pe * dx
    )

    parameter_extraction = ocsm.ParameterExtraction(
        coarse_model, J_pe, y_pe, u_pe, config=config
    )
    parameter_extraction._solve()
    control_pe = u_pe.vector()[:]
    assert (
        np.max(np.abs(control_ocp - control_pe)) / np.max(np.abs(control_ocp)) < 1e-10
    )


def test_parameter_extraction_multiple():
    config = cashocs.load_config(dir_path + "/config_ocp.ini")
    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(10)
    V = FunctionSpace(mesh, "CG", 1)

    y = Function(V)
    z = Function(V)
    p = Function(V)
    q = Function(V)
    u = Function(V)
    v = Function(V)

    states = [y, z]
    adjoints = [p, q]
    controls = [u, v]

    e_y = inner(grad(y), grad(p)) * dx - u * p * dx
    e_z = inner(grad(z), grad(q)) * dx - (y + v) * q * dx

    e = [e_y, e_z]

    bcs1 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])
    bcs2 = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])

    bcs_list = [bcs1, bcs2]

    y_d = Expression("sin(2*pi*x[0])*sin(2*pi*x[1])", degree=1)
    z_d = Expression("sin(4*pi*x[0])*sin(4*pi*x[1])", degree=1)
    alpha = 1e-6
    beta = 1e-4
    J = (
        Constant(0.5) * (y - y_d) * (y - y_d) * dx
        + Constant(0.5) * (z - z_d) * (z - z_d) * dx
        + Constant(0.5 * alpha) * u * u * dx
        + Constant(0.5 * beta) * v * v * dx
    )

    ocp = cashocs.OptimalControlProblem(
        e, bcs_list, J, states, controls, adjoints, config
    )
    ocp.solve()
    u_ocp = u.vector()[:]
    v_ocp = v.vector()[:]

    u.vector()[:] = 0.0
    v.vector()[:] = 0.0
    coarse_model = ocsm.CoarseModel(e, bcs_list, J, states, controls, adjoints, config)
    coarse_model.optimize()
    u_cm = u.vector()[:]
    v_cm = v.vector()[:]

    assert np.max(np.abs(u_ocp - u_cm)) / np.max(np.abs(u_ocp)) < 1e-14
    assert np.max(np.abs(v_ocp - v_cm)) / np.max(np.abs(v_ocp)) < 1e-14

    u_pe = Function(V)
    v_pe = Function(V)
    controls_pe = [u_pe, v_pe]
    y_pe = Function(V)
    z_pe = Function(V)
    states_pe = [y_pe, z_pe]
    J = (
        Constant(0.5) * (y_pe - y_d) * (y_pe - y_d) * dx
        + Constant(0.5) * (z_pe - z_d) * (z_pe - z_d) * dx
        + Constant(0.5 * alpha) * u_pe * u_pe * dx
        + Constant(0.5 * beta) * v_pe * v_pe * dx
    )

    parameter_extraction = ocsm.ParameterExtraction(
        coarse_model, J, states_pe, controls_pe, config
    )
    parameter_extraction._solve()

    assert np.max(np.abs(u_ocp - u_pe.vector()[:])) / np.max(np.abs(u_ocp)) < 1e-10
    assert np.max(np.abs(v_ocp - v_pe.vector()[:])) / np.max(np.abs(v_ocp)) < 1e-10
