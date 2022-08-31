# Copyright (C) 2020-2022 Sebastian Blauth
#
# This file is part of cashocs.
#
# cashocs is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# cashocs is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with cashocs.  If not, see <https://www.gnu.org/licenses/>.

import pathlib
import subprocess
import sys

import pytest

path = pathlib.Path(__file__).resolve().parent

demos = []
demo_files = list(path.glob("**/*.py"))
for f in demo_files:
    demos.append((f.parent, f.name))


@pytest.mark.parametrize("path,name", demos)
def test_demos(path, name):
    ret = subprocess.run([sys.executable, name], cwd=str(path), check=True)
    assert ret.returncode == 0
