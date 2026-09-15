# Copyright (C) 2020-2026 Fraunhofer ITWM and Sebastian Blauth
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

"""Generate Sphinx pages for the package's Typer command line applications."""

import ast
from collections.abc import Iterator
import pathlib
import tomllib


def _entry_point_names(project_root: pathlib.Path) -> dict[str, str]:
    """Return packaged command names keyed by their entry point targets."""
    project_file = project_root / "pyproject.toml"
    project = tomllib.loads(project_file.read_text())
    scripts = project.get("project", {}).get("scripts", {})

    return {
        target: command_name
        for command_name, target in scripts.items()
        if target.startswith("cashocs._cli.")
    }


def _typer_app_names(cli_file: pathlib.Path) -> Iterator[str]:
    """Yield Typer app variable names without importing the CLI module."""
    tree = ast.parse(cli_file.read_text(), filename=str(cli_file))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "app"
            for target in node.targets
        ):
            continue
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "Typer"
        ):
            yield "app"


def discover_cli_apps(
    cli_dir: pathlib.Path, project_root: pathlib.Path
) -> Iterator[tuple[str, str, str]]:
    """Yield Typer apps discovered in modules below ``cli_dir``."""
    entry_point_names = _entry_point_names(project_root)
    package_root = project_root / "cashocs"

    for cli_file in sorted(cli_dir.rglob("*.py")):
        if cli_file.name == "__init__.py":
            continue

        module_name = ".".join(
            ("cashocs", *cli_file.relative_to(package_root).with_suffix("").parts)
        )
        command_name = entry_point_names.get(f"{module_name}:app", cli_file.stem)
        for app_name in _typer_app_names(cli_file):
            yield module_name, app_name, command_name


def write_rst_file(
    module: str, app_name: str, command_name: str, output_dir: pathlib.Path
) -> None:
    """Write the generated Sphinx page for one CLI app."""
    rst_path = output_dir / f"{module}.rst"
    file_contents = f"""{command_name}
{"#" * len(command_name)}

.. typer:: {module}:{app_name}
   :prog: {command_name}
   :width: 70
   :preferred: svg
   :make-sections:
   :show-nested:

"""
    rst_path.write_text(file_contents)


def process() -> None:
    """Discover CLI apps and regenerate their Sphinx pages."""
    docs_source = pathlib.Path(__file__).resolve().parent
    project_root = docs_source.parents[1]
    cli_dir = project_root / "cashocs" / "_cli"
    generated_dir = docs_source / "cli" / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    for generated_file in generated_dir.glob("*.rst"):
        generated_file.unlink()

    for module, app_name, command_name in discover_cli_apps(cli_dir, project_root):
        write_rst_file(module, app_name, command_name, generated_dir)


if __name__ == "__main__":
    process()
