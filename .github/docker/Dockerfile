FROM mambaorg/micromamba:1-jammy AS test-env

USER root

RUN apt-get update --fix-missing && \
    apt-get install -y libgl1-mesa-dev ffmpeg libsm6 libxext6 curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN micromamba install -y -n base -c conda-forge \
      fenics=2019 \
      meshio">=5.0.0" \
      pytest">=7.0.0" \
      gmsh">=4.8" \
      "occt<=7.7.0" \
      coverage">=6.1.0" \
      mpich \
      python=3.11 && \
   micromamba clean --all --yes


FROM test-env AS cashocs

SHELL ["/bin/bash", "--login", "-c"]

COPY . /root/cashocs

RUN micromamba activate base && \
    cd /root/cashocs && \
    pip install .[all]

RUN micromamba activate base && \
    cd /root/cashocs && \
    python3 -m pytest -vv tests/
