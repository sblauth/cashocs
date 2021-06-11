#!/bin/bash
# build script for creating the PYPI files

eval "$(conda shell.bash hook)"
conda activate fenics19

cd /p/tv/DISS_Blauth/cashocs
rm -r build
rm -r cashocs.egg-info
rm -r dist
python3 setup.py sdist bdist_wheel
python3 -m twine upload dist/*
conda deactivate
