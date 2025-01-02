# Copyright (C) 2020-2025 Fraunhofer ITWM and Sebastian Blauth
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

import jupytext


def process():
    """Convert light format demo Python files into MyST flavoured markdown and
    ipynb using Jupytext. These files can then be included in Sphinx
    documentation"""
    # Directories to scan
    subdirs = [pathlib.Path("../../demos/documented")]

    # Iterate over subdirectories containing demos
    for subdir in subdirs:
        # Make demo doc directory
        demo_dir = pathlib.Path("./user/demos")
        demo_dir.mkdir(parents=True, exist_ok=True)

        # Process each demo using jupytext/myst
        for demo in subdir.glob("**/demo*.py"):
            # for demo in subdir.glob("**/demo_space_mapping_semilinear_transmission.py"):
            python_demo = jupytext.read(demo)
            myst_text = jupytext.writes(python_demo, fmt="myst")

            # myst-parser does not process blocks with {code-cell}
            myst_text = myst_text.replace("{code-cell}", "python")
            myst_file = (demo_dir / demo.parent.parent.name / demo.name).with_suffix(
                ".md"
            )
            with open(myst_file, "w") as fw:
                fw.write(myst_text)


if __name__ == "__main__":
    process()
