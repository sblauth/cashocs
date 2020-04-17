A collection of python packages for optimization with partial differential equations.

Overview / Contents
-  adpack (optimization package for fenics which generates adjoints automatically)

Installation / Setup

- Install fenics (only guaranteed to work with version 2018.1) (see [the official installation instructions](https://fenicsproject.org/download/) or follow the steps below for an installation using conda

  - [Install conda (or anaconda)](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

  - Clone (e.g.) this environment by running 

```conda env create -f conda_env.yml -n your_environment_name
```
    where you replace 'your_environment_name' by an appropriate name such as 'fenics'

  - Alternatively, use the environment file conda_env.yml to install the packages (or maybe even newer versions of them) via conda-forge and pip


- Run the setup (currently only supprted from the directory of the repository, i.e., the directory of this file) via

```bash setup.sh
```

- Remember to activate your environment before you will be able to run the packages


- Have fun!
