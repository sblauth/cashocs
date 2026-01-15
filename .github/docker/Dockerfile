FROM condaforge/miniforge3

SHELL ["/bin/bash", "-lc"]

RUN conda create -y -n env \
      fenics=2019 \
      meshio">=5.3" \
      pytest \
      gmsh">=4.8" \
      coverage \
      mpich \
      scipy \
      python=3.14 && \
   conda clean --all --yes

WORKDIR /opt/app

COPY . .

RUN conda run --no-capture-output -n env pip install .[all]

RUN conda run --no-capture-output -n env python -m pytest -vv tests/

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "env"]
CMD ["bash"]
