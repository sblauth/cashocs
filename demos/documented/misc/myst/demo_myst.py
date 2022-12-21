# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.6
# ---

# # Test demo
#
# This demo is used to test myst output via jupytext

# We begin by importing numpy and fenics

from fenics import *

# +
import numpy as np

import cashocs

# -

# Next, we generate a mesh with {py:func}`cashocs.regular_mesh`

# +
mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(16)
# -
