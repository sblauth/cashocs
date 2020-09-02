# CASHOCS


CASHOCS is a computational adjoint-based shape optimization and optimal control software for python.

CASHOCS is based on the finite element package [FEniCS](https://fenicsproject.org) and uses its high-level unified form language UFL to treat general PDE constrained optimization problems, in particuarl, shape optimization and optimal control problems.

Note, that we assume that you are (at least somewhat) familiar with PDE constrained optimization and FEniCS. For a introduction to these topics, we can recommend the textbooks

- Optimal Control and general PDE constrained optimization
    - [Hinze, Ulbrich, Ulbrich, and Pinnau, Optimization with PDE Constraints](https://doi.org/10.1007/978-1-4020-8839-1)
    - [Tröltzsch, Optimal Control of Partial Differential Equations](https://doi.org/10.1090/gsm/112)
- Shape Optimization
    - [Delfour and Zolesio, Shapes and Geometries](https://doi.org/10.1137/1.9780898719826)
    - [Sokolowski and Zolesio, Introduction to Shape Optimization](https://doi.org/10.1007/978-3-642-58106-9)
- FEniCS
    - [Logg, Mardal, and Wells, Automated Solution of Differential Equations by the Finite Element Method](https://doi.org/10.1007/978-3-642-23099-8)

However, the [documented demos](temp_url) also give many references either to the underlying theory of PDE constrained optimization or to relevant demos and documentation of FEniCS.


Table of Contents
=================

  * [Installation](#installation)
  * [Usage](#usage)
  * [Testing](#testing)
  * [License](#license)
  * [Contributing](#contributing)
  * [Contact / About](#contact-about)

Installation
============

- First, install [FEniCS](https://fenicsproject.org/download/), version 2019.1. Note, that FEniCS should be compiled with PETSc and petsc4py.

- Then, install [meshio](https://github.com/nschloe/meshio) with a [h5py](https://www.h5py.org) version that matches the hdf5 version used in FEniCS, and [matplotlib](https://matplotlib.org/)


Note, that if you want to have a [conda](https://docs.conda.io/en/latest/index.html) installation, you can simply create a new environment with

    conda create -n NAME -c conda-forge fenics=2019 meshio matplotlib

which automatically installs all prerequisites to get started.

- You might also want to install [GMSH](https://gmsh.info/). CASHOCS does not necessarily need this to function properly, but it is required for the remeshing functionality.

- Clone this repository with git, and run

        pip3 install .

- Alternatively, you can install CASHOCS via the PYPI

        pip3 install cashocs

 You can install the newest (development) version of cashocs with

        pip3 install git+temp_url

Usage
=====

For an introduction to CASHOCS as well as it's API reference, see [here](temp_url). Moreover, we have several [documented demos](temp_url) which explain how to use CASHOCS
using simple examples. The python code of the demos is located in the "demos" folder, and they can easily be acquired by cloning this repository.


Testing
=======

To run the unit tests for this CASHOCS, run

    cd tests
    pytest

from the source / repository root directory.


License
=======

CASHOCS is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

CASHCOS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with CASHOCS. If not, see <http://www.gnu.org/licenses/>.


Contributing
============

Please see [CONTRIBUTING](./CONTRIBUTING.md).

Contact / About
===============

I'm Sebastian Blauth, a PhD student at Fraunhofer ITWM and TU Kaiserslautern,
and I developed this project as part of my PhD. If you have any questions /
suggestions / feedback, etc., you can contact me via
[sebastian.blauth@itwm.fraunhofer.de](mailto:sebastian.blauth@itwm.fraunhofer.de).
