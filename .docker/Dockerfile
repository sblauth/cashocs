FROM ubuntu:20.04 AS test-env

SHELL ["/bin/bash", "--login", "-c"]

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN apt-get update \
  && apt-get install -y libgl1-mesa-dev ffmpeg libsm6 libxext6 wget && rm -rf /var/lib/apt/lists/* \
  && wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  && mkdir /root/.conda \
  && bash Miniconda3-latest-Linux-x86_64.sh -b \
  && rm -f Miniconda3-latest-Linux-x86_64.sh

RUN conda init \
  && source /root/.bashrc \
  && conda create -n cashocs -c conda-forge fenics=2019 meshio">=4.1.0" pytest">=6.0.0" deprecated">=1.2.10" matplotlib-base gmsh">=4.8" python">=3.6" coverage \
  && echo ". /root/miniconda3/etc/profile.d/conda.sh" >> /root/.bashrc \
  && echo "conda activate base" >> /root/.bashrc \
  && conda clean -a -i -p -t -y


FROM test-env AS cashocs

SHELL ["/bin/bash", "--login", "-c"]

COPY . /root/cashocs

RUN source /root/miniconda3/etc/profile.d/conda.sh \
  && conda activate cashocs \
  && cd /root/cashocs \
  && pip install .
