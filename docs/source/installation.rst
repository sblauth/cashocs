Installation
============

- First, install `FEniCS <https://fenicsproject.org/download/>`_, version 2019.1.
  Note, that FEniCS should be compiled with PETSc and petsc4py.

- Then, install `meshio <https://github.com/nschloe/meshio>`_ with a `h5py <https://www.h5py.org>`_
  version that matches the hdf5 version used in FEniCS, and `matplotlib <https://matplotlib.org/>`_

- You might also want to install `GMSH <https://gmsh.info/>`_, version 4.6.0.
  cashocs does not necessarily need this to function properly,
  but it is required for the remeshing functionality.

Note, that if you want to have a `anaconda / miniconda <https://docs.conda.io/en/latest/index.html>`_
installation, you can simply create a new environment with::

    conda create -n NAME -c conda-forge fenics=2019 meshio matplotlib gmsh=4.6

which automatically installs all prerequisites to get started.

- Clone this repository with git, and run::

        pip3 install .

- Alternatively, you can install cashocs via the PYPI::

        pip3 install cashocs

 You can install the newest (development) version of cashocs with::

        pip3 install git+temp_url