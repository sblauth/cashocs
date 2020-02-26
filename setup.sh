#!/bin/bash


echo '### Added by adpack installation routine' >> ~/.bashrc
echo 'PATH=$PATH:'$(pwd) >> ~/.bashrc
echo 'export PATH' >> ~/.bashrc
echo 'PYTHONPATH=$PYTHONPATH:'$(pwd) >> ~/.bashrc
echo 'export PYTHONPATH' >> ~/.bashrc
echo '### end adpack installation routine' >> ~/.bashrc
