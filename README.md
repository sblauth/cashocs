CASHOCS
=========

CASHOCS is a **C**omputational **A**djoint based **SH**ape optimization and **O**ptimal **C**ontrol **S**oftware for python.


Installation
------------

Setup (from ITWM)

- Load the module tv/cashocs

    `module load tv/cashocs`

    and run

    `activate`

- The git repository with the demos can be found under /p/tv/local/cashocs

- If you want to have a custom installation, you can clone the git repository yourself,
  but then you have to modify your PATH and PYTHONPATH (see, e.g., the setup.sh script)


Setup (external)

- Note, that for all commands shown below it is assumed that you run them from
  the location where you cloned this repository to, otherwise you will be missing
  the corresponding files or might even break things!

- Install fenics (only guaranteed to work with version 2019.1), see
  [the official installation instructions](https://fenicsproject.org/download/)
  or follow the steps below for an installation using conda

  - [Install conda (or anaconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

  - Clone (e.g.) this environment by running

    `conda env create -f conda_env.yml -n your_environment_name`

    where you replace 'your_environment_name' by an appropriate name such as 'cashocs'

  - Alternatively, use the environment file conda_env.yml to install the packages
    (or maybe even newer versions of them) via conda-forge and pip

  - If you want to have the mesh-convert and / or remeshing functionality, you should
    install [gmsh](https://gmsh.info) (version 4.6.0) and [meshio](https://pypi.org/project/meshio/4.0.16/) (version 4.0.16).
    For the mesh convert, meshio version has to be at least 4, and for the remeshing
    gmsh needs to be able to write MSH4.1 format, at least. Additionally, gmsh has to
    be able to be called via

    `gmsh`

    from the command line, for remeshing to work.

- Run the setup (currently only supprted from the directory of the repository, i.e.,
  the directory of this file) via

    `bash setup.sh`


Documentation
-------------

The documentation of the project can be found in the docs folder, just open "index.html"
to view them in a web browser. Hower, I recommend using the documented demos in the demos
folder as reference, as they indicate more clearly how to actually use the project.

Alternatively, see the [documentation](./docs/index.html) and the [documented demos](./demos/docs/demos.html).


Testing
-------

To run the unit tests for this CASHOCS, run

    pytest

from the ./tests directory.


License
-------

CASHOCS is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

CASHCOS is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with CASHOCS. If not, see <http://www.gnu.org/licenses/>.


Contributing
------------

CASHOCS is available on gitlab. As this project is part of my PhD, I am currently
the sole developer. Bug reports, feedback, and further suggestions are always
welcome. If you want to implement new features for CASHOCS, please consider
contacting me first.


Contact / About
---------------

I'm Sebastian Blauth, a PhD student at Fraunhofer ITWM and TU Kaiserslautern,
and I developed this project as part of my work. If you have any questions /
suggestions / feedback, etc., you can contact me via
[sebastian.blauth@itwm.fraunhofer.de](mailto:sebastian.blauth@itwm.fraunhofer.de).
