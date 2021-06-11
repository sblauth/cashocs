#!/bin/bash
# build script for generating the docs 

eval "$(conda shell.bash hook)"
conda activate fenics19

cd /p/tv/DISS_Blauth/cashocs/docs
make clean
make html

conda deactivate
