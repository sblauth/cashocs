"""
Created on 21/12/2022, 08.39

@author: blauths
"""

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
