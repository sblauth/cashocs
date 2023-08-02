FROM mambaorg/micromamba:1-jammy AS test-env

USER root

RUN apt-get update --fix-missing && \
    apt-get install -y libgl1-mesa-dev ffmpeg libsm6 libxext6 curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

USER $MAMBA_USER

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

ARG MAMBA_DOCKERFILE_ACTIVATE=1

FROM test-env AS cashocs

COPY --chown=$MAMBA_USER:$MAMBA_USER . /home/mambauser/cashocs

RUN cd /home/mambauser/cashocs && \
    pip install .[all]

RUN cd /home/mambauser/cashocs && \
    python3 -m pytest -vv tests/
