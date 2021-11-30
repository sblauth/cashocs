FROM continuumio/miniconda3:latest AS test-env

RUN apt-get update --fix-missing \
  && apt-get install -y libgl1-mesa-dev ffmpeg libsm6 libxext6 \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

RUN conda create -n cashocs -c conda-forge fenics=2019 meshio">=4.1.0" pytest">=6.0.0" deprecated">=1.2.10" gmsh">=4.8" python=3.9 \
  && conda clean -tipsy


FROM test-env AS cashocs

SHELL ["/bin/bash", "--login", "-c"]

COPY . /root/cashocs

RUN conda activate cashocs \
  && cd /root/cashocs \
  && pip install .
