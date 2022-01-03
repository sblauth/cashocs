# Copyright (C) 2020-2022 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

import os

import cashocs


dir_path = os.path.dirname(os.path.realpath(__file__))


def test_correct_config():
    config = cashocs.load_config(f"{dir_path}/config_ocp.ini")
    config = cashocs.load_config(f"{dir_path}/config_sop.ini")
    config = cashocs.load_config(f"{dir_path}/config_picard.ini")
    config = cashocs.load_config(f"{dir_path}/config_remesh.ini")

    assert 1 == 1
