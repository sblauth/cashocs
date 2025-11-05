.. image:: https://raw.githubusercontent.com/sblauth/cashocs/main/logos/cashocs_banner.jpg
    :width: 800
    :align: center
    :target: https://github.com/sblauth/cashocs

.. image:: https://img.shields.io/pypi/v/cashocs?style=flat-square
    :target: https://pypi.org/project/cashocs/

.. image:: https://img.shields.io/conda/vn/conda-forge/cashocs?style=flat-square
    :target: https://anaconda.org/conda-forge/cashocs

.. image:: https://img.shields.io/pypi/pyversions/cashocs?style=flat-square
    :target: https://pypi.org/project/cashocs/

.. image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.4035939-informational?style=flat-square
   :target: https://doi.org/10.5281/zenodo.4035939

.. image:: https://img.shields.io/pypi/l/cashocs?color=informational&style=flat-square
    :target: https://pypi.org/project/cashocs/

.. image:: https://img.shields.io/pypi/dm/cashocs?color=informational&style=flat-square
    :target: https://pypistats.org/packages/cashocs

|

.. image:: https://img.shields.io/github/actions/workflow/status/sblauth/cashocs/tests.yml?branch=main&label=tests&style=flat-square
   :target: https://github.com/sblauth/cashocs/actions/workflows/tests.yml

.. image:: https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Fsblauth%2Fcashocs%2Fcoverage%2Fendpoint.json&style=flat-square
   :target: https://htmlpreview.github.io/?https://github.com/sblauth/cashocs/blob/coverage/htmlcov/index.html

.. image:: https://readthedocs.org/projects/cashocs/badge/?version=latest&style=flat-square
    :target: https://cashocs.readthedocs.io/en/latest/?badge=latest

|

cashocs is a finite element software for the automated solution of shape optimization and optimal control problems. It is used to solve problems in fluid dynamics and multiphysics contexts. Its name is an acronym for computational adjoint-based shape optimization and optimal control software and the software is written in Python.


.. contents:: :local:

Introduction
============

cashocs is based on the finite element package `FEniCS
<https://fenicsproject.org>`__ and uses its high-level unified form language UFL
to treat general PDE constrained optimization problems, in particular, shape
optimization and optimal control problems.

To get started with cashocs, take a look at `cashocs in a nutshell <https://cashocs.readthedocs.io/en/stable/about/nutshell/>`_ as well as the in-depth `tutorial <https://cashocs.readthedocs.io/en/stable/user/>`_. There, the core concepts and functionalities of cashocs are explained.

An overview over cashocs and its capabilities can be found in `Blauth - cashocs: A Computational, Adjoint-Based
Shape Optimization and Optimal Control Software <https://doi.org/10.1016/j.softx.2020.100646>`_ and `Blauth - Version 2.0 - cashocs: A Computational, Adjoint-Based Shape Optimization and Optimal Control Software <https://doi.org/10.1016/j.softx.2023.101577>`_. Moreover, note that
the full cashocs documentation is available at `<https://cashocs.readthedocs.io>`_.

For some applications and further information about cashocs, we also refer to the website `Fluid Dynamical Shape Optimization with cashocs <https://www.itwm.fraunhofer.de/en/departments/tv/products-and-services/shape-optimization-cashocs-software.html>`_.


.. readme_start_installation

Installation
============

Via conda-forge
---------------

cashocs is available via the conda package manager, and you can install it (to your currently activated environment) with

.. code-block:: bash

    conda install -c conda-forge cashocs

Alternatively, you might want to create a new, clean conda environment with the command

.. code-block:: bash

    conda create -n ENV_NAME -c conda-forge cashocs

where `ENV_NAME` is the desired name of the new environment.

For more information about conda, please take a look at the `conda documentation <https://docs.conda.io/en/latest/>`_.



Manual Installation
-------------------

- First, install `FEniCS <https://fenicsproject.org/download/>`_, version 2019.1.
  Note that FEniCS should be compiled with PETSc and petsc4py. Moreover, note that cashocs is not yet compatible with the new dolfinx, which is currently under development.

- Then, install `meshio <https://github.com/nschloe/meshio>`_, with a `h5py <https://www.h5py.org>`_
  version that matches the HDF5 version used in FEniCS, and `matplotlib <https://matplotlib.org/>`_.
  The version of meshio should be at least 4, but for compatibility it is recommended to use meshio 4.4.

- You might also want to install `Gmsh <https://gmsh.info/>`_, version 4.8 or later.
  cashocs does not necessarily need this to work properly,
  but it is required for the remeshing functionality.

- You can install cashocs via the `PYPI <https://pypi.org/>`_ as follows

  .. code-block:: bash

      pip install cashocs

- You can install the newest (development) version of cashocs with

  .. code-block:: bash

      pip install git+https://github.com/sblauth/cashocs.git

- To get the latest (development) version of cashocs, clone this repository with git and install it with pip

  .. code-block:: bash

      git clone https://github.com/sblauth/cashocs.git
      cd cashocs
      pip install .


.. note::

    To verify that the installation was successful, run the tests for cashocs
    with

    .. code-block:: bash

        python -m pytest tests/

    or simply

    .. code-block:: bash

        pytest tests/

    from the repository root directory. Note that it might take some
    time to perform all of these tests for the very first time, as FEniCS
    compiles the necessary code. However, on subsequent iterations the
    compiled code is retrieved from a cache, so that the tests are singificantly
    faster.


.. readme_end_installation


Usage
=====

The complete cashocs documentation is available here `<https://cashocs.readthedocs.io>`_. For a detailed
introduction, see the `cashocs tutorial <https://cashocs.readthedocs.io/en/stable/user>`_. The python source code
for the demo programs is located inside the "demos" folder.


.. _citing:

Citing
======

If you use cashocs for your research, please cite the following paper

.. code-block:: text

	cashocs: A Computational, Adjoint-Based Shape Optimization and Optimal Control Software
	Sebastian Blauth
	SoftwareX, Volume 13, 2021
	https://doi.org/10.1016/j.softx.2020.100646

or use the following bibtex entry

.. code-block:: bibtex
	
	@Article{Blauth2021cashocs,
	  author   = {Sebastian Blauth},
	  journal  = {SoftwareX},
	  title    = {{cashocs: A Computational, Adjoint-Based Shape Optimization and Optimal Control Software}},
	  year     = {2021},
	  issn     = {2352-7110},
	  pages    = {100646},
	  volume   = {13},
	  doi      = {https://doi.org/10.1016/j.softx.2020.100646},
	  keywords = {PDE constrained optimization, Adjoint approach, Shape optimization, Optimal control},
	}
	
For more details on how to cite cashocs please take a look at `<https://cashocs.readthedocs.io/en/stable/about/citing/>`_.



References for PDE Constrained Optimization and FEniCS
======================================================

.. readme_start_disclaimer

We assume that you are (at least somewhat) familiar with PDE
constrained optimization and FEniCS. For a introduction to these topics,
we can recommend the textbooks

- Optimal Control and general PDE constrained optimization
    - `Hinze, Ulbrich, Ulbrich, and Pinnau - Optimization with PDE Constraints <https://doi.org/10.1007/978-1-4020-8839-1>`_
    - `Tröltzsch - Optimal Control of Partial Differential Equations <https://doi.org/10.1090/gsm/112>`_
- Shape Optimization
    - `Delfour and Zolesio - Shapes and Geometries <https://doi.org/10.1137/1.9780898719826>`_
    - `Sokolowski and Zolesio - Introduction to Shape Optimization <https://doi.org/10.1007/978-3-642-58106-9>`_
- Topology Optimization
    - `Sokolowski and Novotny - Topological Derivatives in Shape Optimization <https://doi.org/10.1007/978-3-642-35245-4>`_
    - `Amstutz - An Introduction to the Topological Derivative <https://doi.org/10.1108/EC-07-2021-0433>`_
- FEniCS
    - `Logg, Mardal, and Wells - Automated Solution of Differential Equations by the Finite Element Method <https://doi.org/10.1007/978-3-642-23099-8>`_
    - `The FEniCS demos <https://fenicsproject.org/olddocs/dolfin/2019.1.0/python/demos.html>`_

.. readme_end_disclaimer

.. readme_start_license
.. _license:

License
=======

cashocs is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

cashocs is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with cashocs.  If not, see `<https://www.gnu.org/licenses/>`_.


.. readme_end_license


.. readme_start_about

Contact / About
===============

I'm `Sebastian Blauth <https://sblauth.github.io/>`_, a researcher at `Fraunhofer ITWM
<https://www.itwm.fraunhofer.de/en.html>`_. I started developing cashocs during my PhD studies and have
further developed and refined it as part of my employment at Fraunhofer ITWM.
If you have any questions / suggestions / feedback, etc., you can contact me
via `sebastian.blauth@itwm.fraunhofer.de
<mailto:sebastian.blauth@itwm.fraunhofer.de>`_. For more information, visit my website at `<https://sblauth.github.io/>`_.

.. readme_end_about
