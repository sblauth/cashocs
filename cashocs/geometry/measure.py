# Copyright (C) 2020-2022 Sebastian Blauth
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

"""Module for extending the measure functionality."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import fenics
from typing_extensions import Literal
import ufl

from cashocs import _exceptions


class _EmptyMeasure(ufl.Measure):
    """Implements an empty measure (e.g. of a null set).

    This is used for automatic measure generation, e.g., if the fixed boundary is empty
    for a shape optimization problem, and is used to avoid case distinctions.

    Examples:
        The code ::

            dm = _EmptyMeasure(dx)
            u*dm

        is equivalent to ::

            Constant(0)*u*dm

        so that ``fenics.assemble(u*dm)`` generates zeros.

    """

    def __init__(self, measure: fenics.Measure) -> None:
        """Initializes self.

        Args:
            measure: The underlying UFL measure.

        """
        super().__init__(measure.integral_type())
        self.measure = measure

    def __rmul__(self, other: ufl.core.expr.Expr) -> ufl.Form:
        """Multiplies the empty measure to the right.

        Args:
            other: A UFL expression to be integrated over an empty measure.

        Returns:
            The resulting UFL form.

        """
        return fenics.Constant(0) * other * self.measure


def generate_measure(
    idx: List[int], measure: fenics.Measure
) -> Union[fenics.Measure, _EmptyMeasure]:
    """Generates a measure based on indices.

    Generates a :py:class:`fenics.MeasureSum` or
    :py:class:`_EmptyMeasure <cashocs.geometry._EmptyMeasure>` object corresponding to
    ``measure`` and the subdomains / boundaries specified in idx. This is a convenient
    shortcut to writing ``dx(1) + dx(2) + dx(3)`` in case many measures are involved.

    Args:
        idx: A list of indices for the boundary / volume markers that define the (new)
            measure.
        measure: The corresponding UFL measure.

    Returns:
        The corresponding sum of the measures or an empty measure.

    Examples:
        Here, we create a wrapper for the surface measure on the top and bottom of
        the unit square::

            import fenics
            import cashocs

            mesh, _, boundaries, dx, ds, _ = cashocs.regular_mesh(25)
            top_bottom_measure = cashocs.geometry.generate_measure([3,4], ds)
            fenics.assemble(1*top_bottom_measure)

    """
    if len(idx) == 0:
        out_measure = _EmptyMeasure(measure)

    else:
        out_measure = measure(idx[0])

        for i in idx[1:]:
            out_measure += measure(i)

    return out_measure


class NamedMeasure(ufl.Measure):
    """A named integration measure, which can use strings for defining subdomains."""

    def __init__(
        self,
        integral_type: Literal["dx", "ds", "dS"],
        domain: Optional[fenics.Mesh] = None,
        subdomain_id: str = "everywhere",
        metadata: Optional[Dict] = None,
        subdomain_data: Optional[fenics.MeshFunction] = None,
        physical_groups: Optional[Dict[str, Dict[str, int]]] = None,
    ) -> None:
        """See base class."""
        super().__init__(
            integral_type,
            domain=domain,
            subdomain_id=subdomain_id,
            metadata=metadata,
            subdomain_data=subdomain_data,
        )
        if physical_groups is None:
            self.physical_groups = {}
        else:
            self.physical_groups = physical_groups

    def __call__(
        self,
        subdomain_id: Any = None,
        metadata: Any = None,
        domain: Any = None,
        subdomain_data: Any = None,
        degree: Any = None,
        scheme: Any = None,
        rule: Any = None,
    ) -> Optional[ufl.Measure]:
        """See base class.

        This implementation also allows strings for subdomain id.
        """
        if isinstance(subdomain_id, int):
            return super().__call__(
                subdomain_id=subdomain_id,
                metadata=metadata,
                domain=domain,
                subdomain_data=subdomain_data,
                degree=degree,
                scheme=scheme,
                rule=rule,
            )

        elif isinstance(subdomain_id, str):
            if (
                subdomain_id in self.physical_groups["dx"]
                and self._integral_type == "cell"
            ):
                integer_id = self.physical_groups["dx"][subdomain_id]
            elif subdomain_id in self.physical_groups[
                "ds"
            ].keys() and self._integral_type in [
                "exterior_facet",
                "interior_facet",
            ]:
                integer_id = self.physical_groups["ds"][subdomain_id]
            else:
                raise _exceptions.InputError(
                    "cashocs.geometry.measure.NamedMeasure", "subdomain_id"
                )

            return super().__call__(
                subdomain_id=integer_id,
                metadata=metadata,
                domain=domain,
                subdomain_data=subdomain_data,
                degree=degree,
                scheme=scheme,
                rule=rule,
            )

        elif isinstance(subdomain_id, (list)) and all(
            isinstance(x, int) for x in subdomain_id
        ):
            return generate_measure(subdomain_id, self)

        return None
