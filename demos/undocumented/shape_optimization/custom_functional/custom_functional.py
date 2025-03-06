"""
Created on 31/01/2023, 10.00

@author: blauths
"""

from fenics import *

import cashocs

config = cashocs.load_config("./config.ini")
mesh, subdomains, boundaries, dx, ds, dS = cashocs.import_mesh("./mesh/mesh.xdmf")

V = FunctionSpace(mesh, "CG", 1)
u = Function(V)
p = Function(V)

F = inner(grad(u), grad(p)) * dx - p * dx
bcs = DirichletBC(V, Constant(0), boundaries, 1)


class MyFunctional(cashocs.Functional):
    def __init__(self, numerator, denominator, target, weight):
        super().__init__()
        self.numerator = numerator
        self.denominator = denominator
        self.target = target

        self.numerator_value = Constant(0.0)
        self.denominator_value = Constant(0.0)
        self.weight = Constant(weight)

    def coefficients(self):
        coeffs1 = self.denominator.coefficients()
        coeffs2 = self.numerator.coefficients()
        coeffs = tuple(set(coeffs1 + coeffs2))

        return coeffs

    def derivative(self, argument, direction):
        deriv = derivative(
            self.weight
            * (self.numerator_value / self.denominator_value - self.target)
            * (
                (1 / self.denominator_value) * self.numerator
                - (self.numerator_value / self.denominator_value**2) * self.denominator
            ),
            argument,
            direction,
        )

        return deriv

    def evaluate(self):
        num_value = assemble(self.numerator)
        self.numerator_value.assign(num_value)

        denom_value = assemble(self.denominator)
        self.denominator_value.assign(denom_value)

        val = (
            self.weight.values()[0]
            / 2.0
            * pow(num_value / denom_value - self.target, 2)
        )
        return val

    def scale(self, scaling_factor):
        self.weight.assign(scaling_factor)

    def update(self):
        num_value = assemble(self.numerator)
        self.numerator_value.assign(num_value)

        denom_value = assemble(self.denominator)
        self.denominator_value.assign(denom_value)


# J models the functional
# J(\Omega) = \left( \frac{\int_\Omega 1 dx}{\int_\Gamma 1 ds} - C \right)^2
J = MyFunctional(Constant(1) * dx, Constant(1) * ds, 0.25, 1.0)

sop = cashocs.ShapeOptimizationProblem(F, bcs, J, u, p, boundaries, config=config)
sop.solve()
