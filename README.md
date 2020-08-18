DESCENDAL
=========

descendal is a continuous adjoint-based optimal control and shape
optimization package for python


Installation
------------

Setup (from ITWM)

- Load the module tv/fenics

    `module load tv/descendal`

    and run

    `activate`

- The git repository with the demos can be found under /p/tv/local/descendal

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

    where you replace 'your_environment_name' by an appropriate name such as 'descendal'

  - Alternatively, use the environment file conda_env.yml to install the packages
    (or maybe even newer versions of them) via conda-forge and pip


- Run the setup (currently only supprted from the directory of the repository, i.e.,
  the directory of this file) via

    `bash setup.sh`

- Remember to activate your environment with the

    `activate`

  command


- Have fun!


Documentation
-------------

The documentation of the project can be found in the docs folder, just open "index.html"
to view them in a web browser. Hower, I recommend using the documented demos in the demos
folder as reference, as they indicate more clearly how to actually use the project.


Contact / About
---------------

I'm Sebastian Blauth, a PhD student at Fraunhofer ITWM and TU Kaiserslautern,
and I developed this project as part of my work. If you have any questions /
suggestions / feedback, etc., you can contact me via
[sebastian.blauth@itwm.fraunhofer.de](mailto:sebastian.blauth@itwm.fraunhofer.de).

As this project is part of my PhD, it has to be adapted to my needs, so use it with care.
Note, that future updates may break some of your code based on this package,
but I try my best that this does not happen.
