A **C**ontinuous **A**djoint-based **O**ptimal control and **S**hape optimization package for **P**ython (caospy)

Overview / Contents
-  caospy (optimization package for fenics which generates adjoints automatically)

Setup (from ITWM)

- Load the module tv/fenics

    `module load tv/fenics/2018.1.0`
    
    and run
    
    `activate`

- The git repository with the demos can be found under /p/tv/local/caospy

- If you want to have a custom installation, you can clone the git repository yourself, but then you have to modify your PATH and PYTHONPATH (see, e.g., the setup.sh script)
    

Installation / Setup (external)

- Note, that for all commands shown below it is assumed that you run them from the location where you cloned this repository to, otherwise you will be missing the corresponding files or might even break things!

- Install fenics (only guaranteed to work with version 2018.1) - see [the official installation instructions](https://fenicsproject.org/download/) or follow the steps below for an installation using conda

  - [Install conda (or anaconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

  - Clone (e.g.) this environment by running 

    `conda env create -f conda_env.yml -n your_environment_name`
    
    where you replace 'your_environment_name' by an appropriate name such as 'fenics'

  - Alternatively, use the environment file conda_env.yml to install the packages (or maybe even newer versions of them) via conda-forge and pip


- Run the setup (currently only supprted from the directory of the repository, i.e., the directory of this file) via

    `bash setup.sh`

- Remember to activate your environment before you will be able to run the packages


- Have fun!
