#!/bin/bash
# test script for cashocs 

eval "$(conda shell.bash hook)"
conda activate fenics19
cd /p/tv/DISS_Blauth/cashocs/tests
pytest
conda deactivate
