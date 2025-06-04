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

import ast
import importlib
import pathlib
import sys
from unittest import mock


def import_from_path(module_name, file_path):
    with mock.patch("sys.modules"):
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    return module


def list_functions_with_return_types(file_path):
    with open(file_path, "r") as file:
        node = ast.parse(file.read(), filename=file_path)

    functions = []
    for item in node.body:
        if isinstance(item, ast.FunctionDef):
            return_type = ast.unparse(item.returns) if item.returns else None
            functions.append((item.name, return_type))

    return functions


def write_rst_file(file: str, func: str, output_dir: pathlib.Path):
    rst_path = output_dir / (
        file.lstrip("../").replace("/", ".").removesuffix(".py") + ".rst"
    )
    module = file.lstrip("../").replace("/", ".").removesuffix(".py")
    mod = import_from_path("test", file)
    parser_function = getattr(mod, func)
    parser = parser_function()

    name = parser.prog

    fileconts = f"""{name}
{"#"*len(name)}

.. argparse::
   :module: {module}
   :func: {func}
   :prog: {name}
"""
    with open(rst_path, "w") as file:
        file.write(fileconts)


def process():
    cli_dir = pathlib.Path("../../cashocs/_cli")

    generated_dir = pathlib.Path("./cli/generated")
    generated_dir.mkdir(parents=True, exist_ok=True)

    argparse_functions = []
    for pyfile in cli_dir.glob("**/*.py"):
        functions = list_functions_with_return_types(pyfile)

        for fun in functions:
            if fun[1] == "argparse.ArgumentParser":
                argparse_functions.append((str(pyfile), fun[0]))

    for fun in argparse_functions:
        file = fun[0]
        function = fun[1]
        write_rst_file(file, function, generated_dir)
        pass


if __name__ == "__main__":
    process()
