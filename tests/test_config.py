"""
Created on 20/12/2021, 15.58

@author: blauths
"""

import os

import cashocs


dir_path = os.path.dirname(os.path.realpath(__file__))


def test_correct_config():
    config = cashocs.load_config(f"{dir_path}/config_ocp.ini")
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config = cashocs.load_config(f"{dir_path}/config_picard.ini")
    config = cashocs.load_config(f"{dir_path}/config_remesh.ini")

    assert 1 == 1
