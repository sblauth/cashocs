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

ENV CONDA_DEFAULT_ENV=env

RUN echo "conda activate env" >> ~/.bashrc

WORKDIR /opt/app

COPY . .

RUN pip install .[all]

RUN python3 -m pytest -vv tests/
