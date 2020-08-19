#!/bin/bash


echo '### Added by cestrel installation routine' >> ~/.bashrc
echo 'PATH=$PATH:'$(pwd) >> ~/.bashrc
echo 'export PATH' >> ~/.bashrc
echo 'PYTHONPATH=$PYTHONPATH:'$(pwd) >> ~/.bashrc
echo 'export PYTHONPATH' >> ~/.bashrc
echo '### end cestrel installation routine' >> ~/.bashrc
