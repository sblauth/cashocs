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

.. image:: https://img.shields.io/codecov/c/gh/sblauth/cashocs?color=brightgreen&style=flat-square
    :target: https://codecov.io/gh/sblauth/cashocs

.. image:: https://img.shields.io/codacy/grade/4debea4be12c495391e1310025851e55?style=flat-square
    :target: https://app.codacy.com/gh/sblauth/cashocs/dashboard?branch=main

.. image:: https://readthedocs.org/projects/cashocs/badge/?version=latest&style=flat-square
    :target: https://cashocs.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square
    :target: https://github.com/psf/black

|

cashocs is a finite element software for the automated solution of shape optimization and optimal control problems. It is used to solve problems in fluid dynamics and multiphysics contexts. Its name is an acronym for computational adjoint-based shape optimization and optimal control software and the software is written in Python.


.. contents:: :local:

Introduction
============

cashocs is based on the finite element package `FEniCS
<https://fenicsproject.org>`__ and uses its high-level unified form language UFL
to treat general PDE constrained optimization problems, in particular, shape
optimization and optimal control problems.

For some applications and further information about cashocs, we also refer to the website `Fluid Dynamical Shape Optimization with cashocs <https://www.itwm.fraunhofer.de/en/departments/tv/products-and-services/shape-optimization-cashocs-software.html>`_.

.. readme_start_disclaimer

Note, that we assume that you are (at least somewhat) familiar with PDE
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

However, the `cashocs tutorial <https://cashocs.readthedocs.io/en/latest/user>`_ also gives many references either
to the underlying theory of PDE constrained optimization or to relevant demos
and documentation of FEniCS.

An overview over cashocs and its capabilities can be found in `Blauth - cashocs: A Computational, Adjoint-Based
Shape Optimization and Optimal Control Software <https://doi.org/10.1016/j.softx.2020.100646>`_. Moreover, note that
the full cashocs documentation is available at `<https://cashocs.readthedocs.io/en/latest>`_.


.. readme_start_installation

Installation
============

Via conda-forge
---------------

cashocs is available via the anaconda package manager, and you can install it
with

.. code-block:: bash

    conda install -c conda-forge cashocs

Alternatively, you might want to create a new, clean conda environment with the
command

.. code-block:: bash

    conda create -n <ENV_NAME> -c conda-forge cashocs

where `<ENV_NAME>` is the desired name of the new environment.

.. note::

    `Gmsh <https://gmsh.info/>`_ is now (starting with release 1.3.2) automatically installed with anaconda.



Manual Installation
-------------------

- First, install `FEniCS <https://fenicsproject.org/download/>`_, version 2019.1.
  Note that FEniCS should be compiled with PETSc and petsc4py.

- Then, install `meshio <https://github.com/nschloe/meshio>`_, with a `h5py <https://www.h5py.org>`_
  version that matches the HDF5 version used in FEniCS, and `matplotlib <https://matplotlib.org/>`_.
  The version of meshio should be at least 4, but for compatibility it is recommended to use meshio 4.4.

- You might also want to install `Gmsh <https://gmsh.info/>`_, version 4.8.
  cashocs does not necessarily need this to work properly,
  but it is required for the remeshing functionality.

.. note::

    If you are having trouble with using the conversion tool cashocs-convert from
    the command line, then you most likely encountered a problem with hdf5 and h5py.
    This can (hopefully) be resolved by following the suggestions from `this thread
    <https://fenicsproject.discourse.group/t/meshio-convert-to-xdmf-from-abaqus-raises-version-error-for-h5py/1480>`_,
    i.e., you should try to install `meshio <https://github.com/nschloe/meshio>`_
    using the command

    .. code-block:: bash

        pip3 install meshio[all] --no-binary=h5py

- You can install cashocs via the `PYPI <https://pypi.org/>`_ as follows

  .. code-block:: bash

      pip3 install cashocs

- You can install the newest (development) version of cashocs with

  .. code-block:: bash

      pip3 install git+https://github.com/sblauth/cashocs.git

- To get the latest (development) version of cashocs, clone this repository with git and install it with pip

  .. code-block:: bash

      git clone https://github.com/sblauth/cashocs.git
      cd cashocs
      pip3 install .


.. note::

    To verify that the installation was successful, run the tests for cashocs
    with

    .. code-block:: bash

        python3 -m pytest tests/

    or simply

    .. code-block:: bash

        pytest tests/

    from the source / repository root directory. Note that it might take some
    time to perform all of these tests for the very first time, as FEniCS
    compiles the necessary code. However, on subsequent iterations the
    compiled code is retrieved from a cache, so that the tests are singificantly
    faster.


.. readme_end_installation


Usage
=====

The complete cashocs documentation is available here `<https://cashocs.readthedocs.io/en/latest>`_. For a detailed
introduction, see the `cashocs tutorial <https://cashocs.readthedocs.io/en/latest/user>`_. The python source code
for the demo programs is located inside the "demos" folder.


.. readme_start_citing
.. _citing:

Citing
======

If you use cashocs for your research, please cite the following papers

.. code-block:: text

	cashocs: A Computational, Adjoint-Based Shape Optimization and Optimal Control Software
	Sebastian Blauth
	SoftwareX, Volume 13, 2021
	https://doi.org/10.1016/j.softx.2020.100646

as well as

.. code-block:: text

	Version 2.0 - cashocs: A Computational, Adjoint-Based Shape Optimization and Optimal Control Software
	Sebastian Blauth
	SoftwareX, Volume 24, 2023
	https://doi.org/10.1016/j.softx.2023.101577


Additionally, if you are using the nonlinear conjugate gradient methods for shape optimization implemented in cashocs, please cite the following paper
	
.. code-block:: text

	Nonlinear Conjugate Gradient Methods for PDE Constrained Shape Optimization Based on Steklov--Poincaré-Type Metrics
	Sebastian Blauth
	SIAM Journal on Optimization, Volume 31, Issue 3, 2021
	https://doi.org/10.1137/20M1367738

If you are using the space mapping methods for shape optimization, please cite the paper

.. code-block:: text

	Space Mapping for PDE Constrained Shape Optimization
	Sebastian Blauth
	SIAM Journal on Optimization, Volume 33, Issue 3, 2023
	https://doi.org/10.1137/22M1515665

and if you are using the topology optimization methods implemented in cashocs, please cite the paper

.. code-block:: text

	Quasi-Newton Methods for Topology Optimization Using a Level-Set Method
	Sebastian Blauth and Kevin Sturm
	Structural and Multidisciplinary Optimization, Volume 66, 2023
	https://doi.org/10.1007/s00158-023-03653-2

	
If you are using BibTeX, you can use the following entries

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

.. code-block:: bibtex

	@Article{Blauth2023Version,
	  author   = {Sebastian Blauth},
	  journal  = {SoftwareX},
	  title    = {{Version 2.0 - cashocs: A Computational, Adjoint-Based Shape Optimization and Optimal Control Software}},
	  year     = {2023},
	  issn     = {2352-7110},
	  pages    = {101577},
	  volume   = {24},
	  doi      = {https://doi.org/10.1016/j.softx.2023.101577},
	  keywords = {PDE constrained optimization, Shape optimization, Topology optimization, Space mapping},
	}


.. code-block:: bibtex

	@Article{Blauth2021Nonlinear,
	  author   = {Sebastian Blauth},
	  journal  = {SIAM J. Optim.},
	  title    = {{Nonlinear Conjugate Gradient Methods for PDE Constrained Shape Optimization Based on Steklov-Poincaré-Type Metrics}},
	  year     = {2021},
	  number   = {3},
	  pages    = {1658--1689},
	  volume   = {31},
	  doi      = {10.1137/20M1367738},
	  fjournal = {SIAM Journal on Optimization},
	}


.. code-block:: bibtex

	@Article{Blauth2023Space,
	  author   = {Blauth, Sebastian},
	  journal  = {SIAM J. Optim.},
	  title    = {{Space Mapping for PDE Constrained Shape Optimization}},
	  year     = {2023},
	  issn     = {1052-6234,1095-7189},
	  number   = {3},
	  pages    = {1707--1733},
	  volume   = {33},
	  doi      = {10.1137/22M1515665},
	  fjournal = {SIAM Journal on Optimization},
	  mrclass  = {49Q10 (35Q93 49M41 65K05)},
	  mrnumber = {4622415},
	}


.. code-block:: bibtex

	@Article{Blauth2023Quasi,
	  author   = {Blauth, Sebastian and Sturm, Kevin},
	  journal  = {Struct. Multidiscip. Optim.},
	  title    = {{Quasi-Newton methods for topology optimization using a level-set method}},
	  year     = {2023},
	  issn     = {1615-147X,1615-1488},
	  number   = {9},
	  pages    = {203},
	  volume   = {66},
	  doi      = {10.1007/s00158-023-03653-2},
	  fjournal = {Structural and Multidisciplinary Optimization},
	  mrclass  = {99-06},
	  mrnumber = {4635978},
	}

.. readme_end_citing


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

I'm `Sebastian Blauth <https://sblauth.github.io/>`_, a scientific employee at `Fraunhofer ITWM
<https://www.itwm.fraunhofer.de/en.html>`_. I have developed this project as part of my PhD thesis.
If you have any questions / suggestions / feedback, etc., you can contact me
via `sebastian.blauth@itwm.fraunhofer.de
<mailto:sebastian.blauth@itwm.fraunhofer.de>`_. For more information, visit my website at `<https://sblauth.github.io/>`_.

.. readme_end_about
