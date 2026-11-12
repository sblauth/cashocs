FROM docker.io/condaforge/miniforge3

SHELL ["/bin/bash", "-lc"]

RUN conda create -y -n env \
      fenics=2019 \
      meshio">=5.3" \
      pytest \
      gmsh">=4.8" \
      coverage \
      mpich \
      scipy \
      typer \
      python=3.15 && \
   conda clean --all --yes

WORKDIR /opt/app

COPY . .

RUN conda run --live-stream -n env pip install .[all]

RUN conda run --live-stream -n env python -m pytest -vv

ARG CACHE_BUST=1
RUN apt-get update && \
    apt-get upgrade -y --no-install-recommends && \
    apt-get autoremove -y && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/share/doc/ && \
    rm -rf /usr/share/man/

ENTRYPOINT ["conda", "run", "--live-stream", "-n", "env"]
CMD ["bash"]
