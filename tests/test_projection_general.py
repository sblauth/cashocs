from fenics import *
import pytest
import numpy as np

import cashocs
from cashocs._optimization.topology_optimization import bisection

rng = np.random.RandomState(300696)


@pytest.fixture
def levelset_function():

    mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(16)

    CG1 = FunctionSpace(mesh, "CG", 1)
    psi_exp = Expression("x[0]-0.5", degree=1)
    psi = Function(CG1)
    psi.vector()[:] = project(psi_exp, CG1).vector()[:]

    return psi


@pytest.mark.parametrize(
    "volume",
    [
        ([rng.rand() / 4. + 0.5, rng.rand() / 4. + 0.75]),
        ([rng.rand() / 4., rng.rand() / 4. + 0.25]),
        ([rng.rand() / 2., rng.rand() / 2. + 0.5]),
        ([rng.rand()]),
    ],
)
def test_projection_method_for_topology_optimization(
    levelset_function, volume
):
    if len(volume) == 1:
        target = volume[0]
    else:
        target = min(max(volume[0], 0.5), volume[1])
    projection = bisection.projection_levelset(levelset_function, volume_restriction=volume)
    projection.project()
    projection_volume = projection.evaluate(0., 0.)
    assert abs(projection_volume - target) < projection.tolerance_bisect
