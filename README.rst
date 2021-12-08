.. image:: https://raw.githubusercontent.com/sblauth/cashocs/master/logo.png
    :width: 800
    :align: center
    :target: https://github.com/sblauth/cashocs

.. image:: https://img.shields.io/pypi/v/cashocs
    :target: https://pypi.org/project/cashocs/

.. image:: https://img.shields.io/conda/vn/conda-forge/cashocs
    :target: https://anaconda.org/conda-forge/cashocs

.. image:: https://img.shields.io/pypi/pyversions/cashocs
    :target: https://pypi.org/project/cashocs/

.. image:: https://img.shields.io/badge/DOI-10.5281%2Fzenodo.4035939-blue
   :target: https://doi.org/10.5281/zenodo.4035939

.. image:: https://img.shields.io/pypi/l/cashocs?color=informational
    :target: https://pypi.org/project/cashocs/

.. image:: https://img.shields.io/pypi/dm/cashocs?color=informational
    :target: https://pypistats.org/packages/cashocs

|

.. image:: https://img.shields.io/github/workflow/status/sblauth/cashocs/ci?label=tests
    :target: https://github.com/sblauth/cashocs/actions/workflows/ci.yml

.. image:: https://img.shields.io/codecov/c/gh/sblauth/cashocs?color=brightgreen
    :target: https://codecov.io/gh/sblauth/cashocs

.. image:: https://img.shields.io/lgtm/grade/python/github/sblauth/cashocs
    :target: https://lgtm.com/projects/g/sblauth/cashocs

.. image:: https://readthedocs.org/projects/cashocs/badge/?version=latest
    :target: https://cashocs.readthedocs.io/en/latest/?badge=latest

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black

|

CASHOCS is a computational adjoint-based shape optimization and optimal control
software for python.

CASHOCS is based on the finite element package `FEniCS
<https://fenicsproject.org>`__ and uses its high-level unified form language UFL
to treat general PDE constrained optimization problems, in particular, shape
optimization and optimal control problems.

.. readme_start_disclaimer

Note, that we assume that you are (at least somewhat) familiar with PDE
constrained optimization and FEniCS. For a introduction to these topics,
we can recommend the textbooks

- Optimal Control and general PDE constrained optimization
    - `Hinze, Ulbrich, Ulbrich, and Pinnau, Optimization with PDE Constraints <https://doi.org/10.1007/978-1-4020-8839-1>`_
    - `Tröltzsch, Optimal Control of Partial Differential Equations <https://doi.org/10.1090/gsm/112>`_
- Shape Optimization
    - `Delfour and Zolesio, Shapes and Geometries <https://doi.org/10.1137/1.9780898719826>`_
    - `Sokolowski and Zolesio, Introduction to Shape Optimization <https://doi.org/10.1007/978-3-642-58106-9>`_
- FEniCS
    - `Logg, Mardal, and Wells, Automated Solution of Differential Equations by the Finite Element Method <https://doi.org/10.1007/978-3-642-23099-8>`_
    - `The FEniCS demos <https://fenicsproject.org/docs/dolfin/latest/python/demos.html>`_

.. readme_end_disclaimer

However, the `CASHOCS tutorial <https://cashocs.readthedocs.io/en/latest/tutorial_index.html>`_ also gives many references either
to the underlying theory of PDE constrained optimization or to relevant demos
and documentation of FEniCS.

An overview over CASHOCS and its capabilities can be found in `Blauth, cashocs: A Computational, Adjoint-Based
Shape Optimization and Optimal Control Software <https://doi.org/10.1016/j.softx.2020.100646>`_. Moreover, note that
the full CASHOCS documentation is available at `<https://cashocs.readthedocs.io/en/latest/index.html>`_.


.. readme_start_installation

Installation
============

Via conda-forge
---------------

CASHOCS is available via the anaconda package manager, and you can install it
with ::

    conda install -c conda-forge cashocs

Alternatively, you might want to create a new, clean conda environment with the
command ::

    conda create -n <ENV_NAME> -c conda-forge cashocs

where <ENV_NAME> is the desired name of the new environment.

.. note::

    `Gmsh <https://gmsh.info/>`_ is now (starting with release 1.3.2) automatically installed with anaconda.



Manual Installation
-------------------

- First, install `FEniCS <https://fenicsproject.org/download/>`__, version 2019.1.
  Note, that FEniCS should be compiled with PETSc and petsc4py.

- Then, install `meshio <https://github.com/nschloe/meshio>`_, with a `h5py <https://www.h5py.org>`_
  version that matches the HDF5 version used in FEniCS, and `matplotlib <https://matplotlib.org/>`_.
  The version of meshio should be at least 4, but for compatibility it is recommended to use meshio 4.4.

- You might also want to install `Gmsh <https://gmsh.info/>`_, version 4.8.
  CASHOCS does not necessarily need this to work properly,
  but it is required for the remeshing functionality.

.. note::

    If you are having trouble with using the conversion tool cashocs-convert from
    the command line, then you most likely encountered a problem with hdf5 and h5py.
    This can (hopefully) be resolved by following the suggestions from `this thread
    <https://fenicsproject.discourse.group/t/meshio-convert-to-xdmf-from-abaqus-raises-version-error-for-h5py/1480>`_,
    i.e., you should try to install `meshio <https://github.com/nschloe/meshio>`_
    using the command ::

        pip3 install meshio[all] --no-binary=h5py

- You can install CASHOCS via the `PYPI <https://pypi.org/>`_::

        pip3 install cashocs

  You can install the newest (development) version of CASHOCS with::

        pip3 install git+https://github.com/sblauth/cashocs.git

- To get the latest (development) version of CASHOCS, clone this repository with git and install it with pip ::

        git clone https://github.com/sblauth/cashocs.git
        cd cashocs
        pip3 install .


.. note::

    To verify that the installation was successful, run the tests for CASHOCS
    with ::

        python3 -m pytest

    or simply ::

        pytest

    from the source / repository root directory. Note, that it might take some
    time to perform all of these tests for the very first time, as FEniCS
    compiles the necessary code. However, on subsequent iterations the
    compiled code is retrieved from a cache, so that the tests are singificantly
    faster.


.. readme_end_installation


Usage
=====

The complete CASHOCS documentation is available here `<https://cashocs.readthedocs.io/en/latest/index.html>`_. For a detailed
introduction, see the `CASHOCS tutorial <https://cashocs.readthedocs.io/en/latest/tutorial_index.html>`_. The python source code
for the demo programs is located inside the "demos" folder.


.. readme_start_citing
.. _citing:

Citing
======

If you use cashocs for your research, I would be grateful if you would cite the following paper ::

	cashocs: A Computational, Adjoint-Based Shape Optimization and Optimal Control Software
	Sebastian Blauth
	SoftwareX, Volume 13, 2021
	https://doi.org/10.1016/j.softx.2020.100646

Additionally, if you are using the nonlinear conjugate gradient methods for shape optimization implemented in cashocs, please cite the following paper ::
	
	Nonlinear Conjugate Gradient Methods for PDE Constrained Shape Optimization Based on Steklov--Poincaré-Type Metrics
	Sebastian Blauth
	SIAM Journal on Optimization, Volume 31, Issue 3, 2021
	https://doi.org/10.1137/20M1367738

	
If you are using BibTeX, you can use the following entries::
	
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

as well as ::	

	@Article{Blauth2021Nonlinear,
		author   = {Sebastian Blauth},
		journal  = {SIAM J. Optim.},
		title    = {{N}onlinear {C}onjugate {G}radient {M}ethods for {PDE} {C}onstrained {S}hape {O}ptimization {B}ased on {S}teklov-{P}oincaré-{T}ype {M}etrics},
		year     = {2021},
		number   = {3},
		pages    = {1658--1689},
		volume   = {31},
		doi      = {10.1137/20M1367738},
		fjournal = {SIAM Journal on Optimization},
	}

.. readme_end_citing


.. readme_start_license
.. _license:

License
=======

CASHOCS is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

CASHOCS is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with CASHOCS.  If not, see `<https://www.gnu.org/licenses/>`_.


.. readme_end_license


.. readme_start_about

Contact / About
===============

I'm `Sebastian Blauth <https://www.itwm.fraunhofer.de/en/departments/tv/staff/sebastian-blauth.html>`_, a scientific employee at `Fraunhofer ITWM
<https://www.itwm.fraunhofer.de/en.html>`_. I have developed this project as part of my PhD thesis.
If you have any questions / suggestions / feedback, etc., you can contact me
via `sebastian.blauth@itwm.fraunhofer.de
<mailto:sebastian.blauth@itwm.fraunhofer.de>`_ or `sebastianblauth@web.de
<mailto:sebastianblauth@web.de>`_.

.. readme_end_about
