# Copyright (C) 2020-2023 Sebastian Blauth
#
# This file is part of cashocs.
#
# cashocs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cashocs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cashocs.  If not, see <https://www.gnu.org/licenses/>.

import numpy as np
import cashocs
from cashocs._optimization.shape_optimization.mesh_constraints import ConstraintManager
import pathlib
from fenics import *

dir_path = str(pathlib.Path(__file__).parent)


def compute_convergence_rates(
    epsilons: list[float], residuals: list[float], verbose: bool = True
) -> list[float]:
    """Computes the convergence rate of the Taylor test.

    Args:
        epsilons: The step sizes.
        residuals: The corresponding residuals.
        verbose: Prints the result to the console, if ``True``. Default is ``True``.

    Returns:
        The computed convergence rates

    """
    rates: list[float] = []
    for i in range(1, len(epsilons)):
        rate: float = np.log(residuals[i] / residuals[i - 1]) / np.log(
            epsilons[i] / epsilons[i - 1]
        )
        rates.append(rate)

    print(f"Taylor test convergence rate: {rates}", flush=True)

    return rates


def test_triangle_mesh_constraints():
    cfg = cashocs.io.config.Config()
    cfg.set("MeshQualityConstraints", "min_angle", "30.0")
    cfg.set("MeshQualityConstraints", "tol", "1e-2")
    cfg.set("ShapeGradient", "shape_bdry_fix", "[]")
    cfg.set("ShapeGradient", "shape_bdry_def", "[1,2,3,4]")

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(8)
    bbtree = mesh.bounding_box_tree()
    rng = np.random.RandomState(300696)
    VCG = VectorFunctionSpace(mesh, "CG", 1)

    defo = 0.01 * rng.standard_normal(mesh.coordinates().shape)
    mesh.coordinates()[:, :] += defo
    bbtree.build(mesh)

    cm = ConstraintManager(cfg, mesh, boundaries, VCG)
    x = mesh.coordinates().copy().reshape(-1)

    residuals = []
    epsilons = [1e-4 / 2**i for i in range(6)]
    h = rng.standard_normal(x.shape)
    f_k = cm.evaluate(x)
    gradient = cm.compute_gradient(x)
    directional_derivative = gradient @ h

    for eps in epsilons:
        x_mod = x + eps * h[cm.v2d]
        f_mod = cm.evaluate(x_mod)

        res = abs(f_mod - f_k - eps * directional_derivative)
        residuals.append(res)

    rates = np.array(compute_convergence_rates(epsilons, residuals)).T
    assert np.min(rates) > 1.8
