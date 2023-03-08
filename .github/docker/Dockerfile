FROM continuumio/miniconda3:latest AS test-env

RUN apt-get update --fix-missing \
  && apt-get install -y libgl1-mesa-dev ffmpeg libsm6 libxext6 curl \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN conda install -c conda-forge mamba \
  && mamba create -n cashocs -c conda-forge fenics=2019 meshio">=5.0.0" pytest">=7.0.0" gmsh">=4.8" coverage">=6.1.0" mpich imath"<=3.1.6" python=3.11 \
  && conda clean --all --yes


FROM test-env AS cashocs

SHELL ["/bin/bash", "--login", "-c"]

COPY . /root/cashocs

RUN conda activate cashocs \
  && cd /root/cashocs \
  && pip install .[all]

RUN conda activate cashocs \
  && cd /root/cashocs \
  && python3 -m pytest -vv tests/
