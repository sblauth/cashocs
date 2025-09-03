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

"""Management of configuration files."""

from __future__ import annotations

from configparser import ConfigParser
import json
import pathlib
from typing import Any

from cashocs import _exceptions

try:
    import cashocs_extensions  # pylint: disable=unused-import

    has_cashocs_extensions = True
except ImportError:
    has_cashocs_extensions = False


def load_config(path: str) -> ConfigParser:
    """Loads a config object from a config file.

    Loads the config from a .ini file via the configparser package.

    Args:
        path: The path to the .ini file storing the configuration.

    Returns:
        The output config file, which includes the path to the .ini file.

    """
    return Config(path)


def _check_for_config_list(string: str) -> bool:
    """Checks, if string is a valid python list consisting of numbers.

    Args:
        string: The input string.

    Returns:
        ``True`` if the string is valid, ``False`` otherwise

    """
    result = False

    for char in string:
        if not (
            char.isdigit()
            or char.isalpha()
            or char.isspace()
            or char in ["[", "]", ".", ",", "-", '"', "'", "_"]
        ):
            return result

    if string[0] != "[":
        return result
    if string[-1] != "]":
        return result

    return True


class Config(ConfigParser):
    """Class for handling the config in cashocs."""

    def __init__(self, config_file: str | None = None) -> None:
        """Initializes self.

        Args:
            config_file: Path to the config file.

        """
        super().__init__()
        self.config_errors: list[str] = []

        self.config_scheme: dict[str, dict[str, dict[str, Any]]] = {
            "Mesh": {
                "gmsh_file": {
                    "type": "str",
                    "attributes": ["file"],
                    "file_extension": "msh",
                },
                "geo_file": {
                    "type": "str",
                    "attributes": ["file"],
                    "file_extension": "geo",
                },
                "remesh": {
                    "type": "bool",
                    "requires": [("Mesh", "gmsh_file")],
                },
                "show_gmsh_output": {
                    "type": "bool",
                },
            },
            "StateSystem": {
                "is_linear": {
                    "type": "bool",
                },
                "newton_rtol": {
                    "type": "float",
                    "attributes": ["less_than_one", "positive"],
                },
                "newton_atol": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "newton_iter": {
                    "type": "int",
                    "attributes": ["non_negative"],
                },
                "newton_damped": {
                    "type": "bool",
                },
                "newton_inexact": {
                    "type": "bool",
                },
                "newton_verbose": {
                    "type": "bool",
                },
                "picard_iteration": {
                    "type": "bool",
                },
                "picard_rtol": {
                    "type": "float",
                    "attributes": ["positive", "less_than_one"],
                },
                "picard_atol": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "picard_iter": {
                    "type": "int",
                    "attributes": ["non_negative"],
                },
                "picard_verbose": {
                    "type": "bool",
                },
                "backend": {
                    "type": "str",
                    "possible_options": ["cashocs", "petsc"],
                },
                "use_adjoint_linearizations": {
                    "type": "bool",
                },
            },
            "OptimizationRoutine": {
                "algorithm": {
                    "type": "str",
                    "possible_options": [
                        "gd",
                        "gradient_descent",
                        "bfgs",
                        "lbfgs",
                        "nonlinear_cg",
                        "ncg",
                        "nonlinear_conjugate_gradient",
                        "conjugate_gradient",
                        "newton",
                        "sphere_combination",
                        "convex_combination",
                        "none",
                    ],
                },
                "rtol": {
                    "type": "float",
                    "attributes": ["less_than_one", "non_negative"],
                },
                "atol": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "max_iter": {
                    "type": "int",
                    "attributes": ["non_negative"],
                },
                "gradient_method": {
                    "type": "str",
                    "possible_options": ["direct", "iterative"],
                },
                "gradient_tol": {
                    "type": "float",
                    "attributes": ["less_than_one", "positive"],
                },
                "soft_exit": {
                    "type": "bool",
                },
            },
            "LineSearch": {
                "method": {
                    "type": "str",
                    "possible_options": ["armijo", "polynomial"],
                },
                "initial_stepsize": {
                    "type": "float",
                    "attributes": ["positive"],
                },
                "epsilon_armijo": {
                    "type": "float",
                    "attributes": ["positive", "less_than_one"],
                },
                "beta_armijo": {
                    "type": "float",
                    "attributes": ["positive", "larger_than_one"],
                },
                "safeguard_stepsize": {"type": "bool"},
                "polynomial_model": {
                    "type": "str",
                    "possible_options": ["cubic", "quadratic"],
                },
                "factor_high": {
                    "type": "float",
                    "attributes": ["less_than_one", "positive"],
                    "larger_than": ("LineSearch", "factor_low"),
                },
                "factor_low": {
                    "type": "float",
                    "attributes": ["less_than_one", "positive"],
                },
                "fail_if_not_converged": {
                    "type": "bool",
                },
            },
            "AlgoLBFGS": {
                "bfgs_memory_size": {
                    "type": "int",
                    "attributes": ["non_negative"],
                },
                "use_bfgs_scaling": {
                    "type": "bool",
                },
                "bfgs_periodic_restart": {
                    "type": "int",
                    "attributes": ["non_negative"],
                },
                "damped": {
                    "type": "bool",
                },
            },
            "AlgoCG": {
                "cg_method": {
                    "type": "str",
                    "possible_options": ["fr", "pr", "hs", "dy", "hz"],
                },
                "cg_periodic_restart": {
                    "type": "bool",
                },
                "cg_periodic_its": {
                    "type": "int",
                    "attributes": ["non_negative"],
                },
                "cg_relative_restart": {
                    "type": "bool",
                },
                "cg_restart_tol": {
                    "type": "float",
                    "attributes": ["positive"],
                },
            },
            "AlgoTNM": {
                "inner_newton": {
                    "type": "str",
                    "possible_options": ["cg", "cr"],
                },
                "inner_newton_rtol": {
                    "type": "float",
                    "attributes": ["positive", "less_than_one"],
                },
                "inner_newton_atol": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "max_it_inner_newton": {
                    "type": "int",
                    "attributes": ["non_negative"],
                },
            },
            "ShapeGradient": {
                "shape_bdry_def": {
                    "type": "list",
                },
                "shape_bdry_fix": {
                    "type": "list",
                },
                "shape_bdry_fix_x": {
                    "type": "list",
                },
                "shape_bdry_fix_y": {
                    "type": "list",
                },
                "shape_bdry_fix_z": {
                    "type": "list",
                },
                "fixed_dimensions": {
                    "type": "list",
                },
                "use_pull_back": {
                    "type": "bool",
                },
                "lambda_lame": {
                    "type": "float",
                },
                "damping_factor": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "mu_def": {
                    "type": "float",
                    "attributes": ["positive"],
                },
                "mu_fix": {
                    "type": "float",
                    "attributes": ["positive"],
                },
                "use_sqrt_mu": {
                    "type": "bool",
                },
                "inhomogeneous": {
                    "type": "bool",
                },
                "update_inhomogeneous": {
                    "type": "bool",
                },
                "inhomogeneous_exponent": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "use_distance_mu": {
                    "type": "bool",
                },
                "dist_min": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "dist_max": {
                    "type": "float",
                    "larger_equal_than": ("ShapeGradient", "dist_min"),
                    "attributes": ["non_negative"],
                },
                "mu_min": {
                    "type": "float",
                    "attributes": ["positive"],
                },
                "mu_max": {
                    "type": "float",
                    "attributes": ["positive"],
                },
                "boundaries_dist": {
                    "type": "list",
                },
                "distance_method": {
                    "type": "str",
                    "possible_options": [
                        "eikonal",
                        "poisson",
                    ],
                },
                "smooth_mu": {
                    "type": "bool",
                },
                "use_p_laplacian": {
                    "type": "bool",
                },
                "p_laplacian_power": {
                    "type": "int",
                    "attributes": ["larger_than_one"],
                },
                "p_laplacian_stabilization": {
                    "type": "float",
                    "attributes": ["non_negative", "less_than_one"],
                },
                "degree_estimation": {
                    "type": "bool",
                },
                "global_deformation": {
                    "type": "bool",
                },
                "test_for_intersections": {
                    "type": "bool",
                },
                "reextend_from_boundary": {
                    "type": "bool",
                },
                "reextension_mode": {
                    "type": "str",
                    "possible_options": ["surface", "normal"],
                },
            },
            "Regularization": {
                "factor_volume": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "target_volume": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "use_initial_volume": {
                    "type": "bool",
                },
                "factor_surface": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "target_surface": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "use_initial_surface": {
                    "type": "bool",
                },
                "factor_curvature": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "factor_barycenter": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "target_barycenter": {
                    "type": "list",
                },
                "use_initial_barycenter": {
                    "type": "bool",
                },
                "x_start": {
                    "type": "float",
                },
                "x_end": {
                    "type": "float",
                    "larger_than": ("Regularization", "x_start"),
                },
                "y_start": {
                    "type": "float",
                },
                "y_end": {
                    "type": "float",
                    "larger_than": ("Regularization", "y_start"),
                },
                "z_start": {
                    "type": "float",
                },
                "z_end": {
                    "type": "float",
                    "larger_than": ("Regularization", "z_start"),
                },
                "use_relative_scaling": {
                    "type": "bool",
                },
            },
            "MeshQuality": {
                "volume_change": {
                    "type": "float",
                    "attributes": ["positive", "larger_than_one"],
                },
                "angle_change": {
                    "type": "float",
                    "attributes": ["positive"],
                },
                "tol_lower": {
                    "type": "float",
                    "attributes": ["less_than_one", "non_negative"],
                },
                "tol_upper": {
                    "type": "float",
                    "attributes": ["less_than_one", "positive"],
                    "larger_than": ("MeshQuality", "tol_lower"),
                },
                "measure": {
                    "type": "str",
                    "possible_options": [
                        "skewness",
                        "radius_ratios",
                        "maximum_angle",
                        "condition_number",
                    ],
                },
                "type": {
                    "type": "str",
                    "possible_options": [
                        "min",
                        "avg",
                        "q",
                        "minimum",
                        "average",
                        "quantile",
                    ],
                },
                "quantile": {
                    "type": "float",
                    "attributes": ["non_negative", "less_than_one"],
                },
                "remesh_iter": {
                    "type": "int",
                    "attributes": ["non_negative"],
                },
            },
            "TopologyOptimization": {
                "angle_tol": {
                    "type": "float",
                    "attributes": ["positive"],
                },
                "interpolation_scheme": {
                    "type": "str",
                    "possible_options": ["angle", "volume"],
                },
                "normalize_topological_derivative": {
                    "type": "bool",
                },
                "re_normalize_levelset": {
                    "type": "bool",
                },
                "topological_derivative_is_identical": {
                    "type": "bool",
                },
                "tol_bisection": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
                "max_iter_bisection": {
                    "type": "int",
                    "attributes": ["non_negative"],
                },
            },
            "Output": {
                "verbose": {
                    "type": "bool",
                },
                "save_results": {
                    "type": "bool",
                },
                "save_txt": {
                    "type": "bool",
                },
                "save_state": {
                    "type": "bool",
                },
                "save_adjoint": {
                    "type": "bool",
                },
                "save_gradient": {
                    "type": "bool",
                },
                "save_mesh": {
                    "type": "bool",
                    "requires": [("Mesh", "gmsh_file")],
                },
                "result_dir": {
                    "type": "str",
                },
                "precision": {
                    "type": "int",
                    "attributes": ["positive"],
                },
                "time_suffix": {
                    "type": "bool",
                },
            },
            "Debug": {
                "remeshing": {
                    "type": "bool",
                },
                "restart": {
                    "type": "bool",
                },
            },
            "DEFAULT": {},
        }
        self.default_config_str = """
[Mesh]
remesh = False
show_gmsh_output = False

[StateSystem]
is_linear = False
newton_rtol = 1e-11
newton_atol = 1e-13
newton_iter = 50
newton_damped = False
newton_inexact = False
newton_verbose = False
picard_iteration = False
picard_rtol = 1e-10
picard_atol = 1e-12
picard_iter = 50
picard_verbose = False
backend = cashocs
use_adjoint_linearizations = False

[OptimizationRoutine]
algorithm = none
rtol = 1e-3
atol = 0.0
max_iter = 100
soft_exit = False
gradient_tol = 1e-9
gradient_method = direct

[LineSearch]
method = armijo
epsilon_armijo = 1e-4
beta_armijo = 2.0
initial_stepsize = 1.0
safeguard_stepsize = True
polynomial_model = cubic
factor_high = 0.5
factor_low = 0.1
fail_if_not_converged = False

[ShapeGradient]
lambda_lame = 0.0
damping_factor = 0.0
mu_def = 1.0
mu_fix = 1.0
use_sqrt_mu = False
use_p_laplacian = False
p_laplacian_power = 2
p_laplacian_stabilization = 0.0
use_pull_back = True
use_distance_mu = False
mu_min = 1.0
mu_max = 1.0
dist_min = 1.0
dist_max = 1.0
boundaries_dist = []
distance_method = eikonal
smooth_mu = False
inhomogeneous = False
update_inhomogeneous = False
inhomogeneous_exponent = 1.0
fixed_dimensions = []
shape_bdry_def = []
shape_bdry_fix = []
shape_bdry_fix_x = []
shape_bdry_fix_y = []
shape_bdry_fix_z = []
degree_estimation = True
global_deformation = False
test_for_intersections = True
reextend_from_boundary = False
reextension_mode = surface

[Regularization]
factor_volume = 0.0
target_volume = 0.0
use_initial_volume = False
factor_surface = 0.0
target_surface = 0.0
use_initial_surface = False
factor_curvature = 0.0
factor_barycenter = 0.0
target_barycenter = [0.0, 0.0, 0.0]
use_initial_barycenter = False
use_relative_scaling = False
x_start = 0.0
x_end = 1.0
y_start = 0.0
y_end = 1.0
z_start = 0.0
z_end = 1.0

[AlgoTNM]
inner_newton = cr
max_it_inner_newton = 50
inner_newton_rtol = 1e-15
inner_newton_atol = 0.0

[AlgoLBFGS]
bfgs_memory_size = 5
use_bfgs_scaling = True
bfgs_periodic_restart = 0
damped = False

[AlgoCG]
cg_method = DY
cg_periodic_restart = False
cg_periodic_its = 10
cg_relative_restart = False
cg_restart_tol = 0.25

[MeshQuality]
tol_lower = 0.0
tol_upper = 1e-15
measure = skewness
type = min
quantile = 0.0
volume_change = inf
angle_change = inf
remesh_iter = 0

[TopologyOptimization]
angle_tol = 1.0
interpolation_scheme = volume
normalize_topological_derivative = False
re_normalize_levelset = False
topological_derivative_is_identical = False
tol_bisection = 1e-4
max_iter_bisection = 100

[Output]
save_results = True
verbose = False
save_txt = False
save_state = False
save_adjoint = False
save_gradient = False
save_mesh = False
result_dir = ./results
precision = 3
time_suffix = False

[Debug]
remeshing = False
restart = False
"""

        self.read_string(self.default_config_str)

        if has_cashocs_extensions:
            self.config_scheme.update(cashocs_extensions.config.config_scheme)
            self.read_string(cashocs_extensions.config.default_config_str)

        if config_file is not None:
            file = pathlib.Path(config_file)
            if file.is_file():
                self.read(config_file)
            else:
                raise _exceptions.InputError(
                    "cashocs.Config",
                    "config_file",
                    f"Could not find the specified config file {config_file}. "
                    "Please supply a path to an existing configuration file.",
                )

    def getlist(self, section: str, option: str, **kwargs: Any) -> list:
        """Extracts a list from a config file.

        Args:
            section: The section where the list is placed.
            option: The option which contains the list.
            **kwargs: A list of keyword arguments that get passed to
                :py:meth:``self.get``

        Returns:
            The list which is specified in section ``section`` and key ``option``.

        """
        if (
            self.config_scheme[section][option]["type"] == "list"
        ) and _check_for_config_list(self.get(section, option)):
            py_list: list = json.loads(self.get(section, option, **kwargs))
            return py_list
        else:
            raise _exceptions.InputError(
                "Config.getlist",
                "option",
                f"option {option} in section {section} cannot be used as list.",
            )

    def validate_config(self) -> None:
        """Validates the configuration file."""
        self._check_sections()
        self._check_keys()

        if len(self.config_errors) > 0:
            raise _exceptions.ConfigError(self.config_errors)

    def _check_sections(self) -> None:
        """Checks whether all sections are valid."""
        for section_name, section in self.items():
            if section_name not in self.config_scheme:
                self.config_errors.append(
                    f"The following section is not valid: {section}\n"
                )

    def _check_keys(self) -> None:
        """Checks the keys of the sections."""
        for section_name, section in self.items():
            for key in section.keys():
                if section_name in self.config_scheme:
                    if key not in self.config_scheme[section_name].keys():
                        self.config_errors.append(
                            f"Key {key} is not valid for section {section_name}.\n"
                        )
                    else:
                        self._check_key_type(section_name, key)
                        self._check_possible_options(section_name, key)
                        self._check_attributes(section_name, key)
                        self._check_key_requirements(section_name, key)
                        self._check_larger_than_relation(section_name, key)
                        self._check_larger_equal_than_relation(section_name, key)

    def _check_key_type(self, section: str, key: str) -> None:
        """Checks if the type of the key is correct.

        Args:
            section: The corresponding section
            key: The corresponding key

        """
        key_type = self.config_scheme[section][key]["type"]
        try:
            if key_type.casefold() == "str":
                self.get(section, key)
            elif key_type.casefold() == "bool":
                self.getboolean(section, key)
            elif key_type.casefold() == "int":
                self.getint(section, key)
            elif key_type.casefold() == "float":
                self.getfloat(section, key)
            elif key_type.casefold() == "list":
                if not _check_for_config_list(self.get(section, key)):
                    raise ValueError
        except ValueError:
            self.config_errors.append(
                f"Key {key} in section {section} has the wrong type. "
                f"Required type is {key_type}.\n"
            )

    def _check_key_requirements(self, section: str, key: str) -> None:
        """Checks, whether the requirements for the key are satisfied.

        Args:
            section: The corresponding section
            key: The corresponding key

        """
        if (
            self.config_scheme[section][key]["type"].casefold() == "bool"
            and self[section][key].casefold() == "true"
        ):
            if "requires" in self.config_scheme[section][key].keys():
                requirements = self.config_scheme[section][key]["requires"]
                for req in requirements:
                    if not self.has_option(req[0], req[1]):
                        self.config_errors.append(
                            f"Key {key} in section {section} requires "
                            f"key {req[1]} in section {req[0]} to be present.\n"
                        )

    def _check_possible_options(self, section: str, key: str) -> None:
        """Checks, whether the given option is possible.

        Args:
            section: The corresponding section
            key: The corresponding key

        """
        if "possible_options" in self.config_scheme[section][key].keys():
            if (
                self[section][key].casefold()
                not in self.config_scheme[section][key]["possible_options"]
            ):
                self.config_errors.append(
                    f"Key {key} in section {section} has a wrong value. "
                    f"Possible options are "
                    f"{self.config_scheme[section][key]['possible_options']}.\n"
                )

    def _check_larger_than_relation(self, section: str, key: str) -> None:
        """Checks, whether a given option is larger than one.

        Args:
            section: The corresponding section
            key: The corresponding key

        """
        if "larger_than" in self.config_scheme[section][key].keys():
            higher_value = self.getfloat(section, key)
            partner = self.config_scheme[section][key]["larger_than"]
            lower_value = self.getfloat(partner[0], partner[1])
            if lower_value >= higher_value:
                self.config_errors.append(
                    f"The value of key {key} in section {section} is smaller than "
                    f"the value of key {partner[1]} in section {partner[0]}, "
                    f"but it should be larger.\n"
                )

    def _check_larger_equal_than_relation(self, section: str, key: str) -> None:
        """Checks, whether a given option is larger or equal to another.

        Args:
            section: The corresponding section
            key: The corresponding key

        """
        if "larger_equal_than" in self.config_scheme[section][key].keys():
            higher_value = self.getfloat(section, key)
            partner = self.config_scheme[section][key]["larger_equal_than"]
            lower_value = self.getfloat(partner[0], partner[1])
            if lower_value > higher_value:
                self.config_errors.append(
                    f"The value of key {key} in section {section} is smaller than "
                    f"the value of key {partner[1]} in section {partner[0]}, "
                    f"but it should be larger.\n"
                )

    def _check_attributes(self, section: str, key: str) -> None:
        """Checks the attributes of a key.

        Args:
            section: The corresponding section
            key: The corresponding key

        """
        if "attributes" in self.config_scheme[section][key].keys():
            key_attributes = self.config_scheme[section][key]["attributes"]
            self._check_file_attribute(section, key, key_attributes)
            self._check_non_negative_attribute(section, key, key_attributes)
            self._check_positive_attribute(section, key, key_attributes)
            self._check_less_than_one_attribute(section, key, key_attributes)
            self._check_larger_than_one_attribute(section, key, key_attributes)

    def _check_file_attribute(
        self, section: str, key: str, key_attributes: list[str]
    ) -> None:
        """Checks, whether a file specified in key exists.

        Args:
            section: The corresponding section
            key: The corresponding key
            key_attributes: The list of attributes for key.

        """
        if "file" in key_attributes:
            file = pathlib.Path(self.get(section, key))
            if not file.is_file():
                self.config_errors.append(
                    f"Key {key} in section {section} should point to a file, "
                    f"but the file does not exist.\n"
                )

            self._check_file_extension(
                section, key, self.config_scheme[section][key]["file_extension"]
            )

    def _check_file_extension(self, section: str, key: str, extension: str) -> None:
        """Checks, whether key has the correct file extension.

        Args:
            section: The corresponding section.
            key: The corresponding key.
            extension: The file extension.

        """
        path_to_file = self.get(section, key)
        if not path_to_file.split(".")[-1] == extension:
            self.config_errors.append(
                f"Key {key} in section {section} has the wrong file extension, "
                f"it should end in .{extension}.\n"
            )

    def _check_non_negative_attribute(
        self, section: str, key: str, key_attributes: list[str]
    ) -> None:
        """Checks, whether key is nonnegative.

        Args:
            section: The corresponding section
            key: The corresponding key
            key_attributes: The list of attributes for key.

        """
        if "non_negative" in key_attributes:
            if self.getfloat(section, key) < 0:
                self.config_errors.append(
                    f"Key {key} in section {section} is negative, but it must not be.\n"
                )

    def _check_positive_attribute(
        self, section: str, key: str, key_attributes: list[str]
    ) -> None:
        """Checks, whether key is positive.

        Args:
            section: The corresponding section
            key: The corresponding key
            key_attributes: The list of attributes for key.

        """
        if "positive" in key_attributes:
            if self.getfloat(section, key) <= 0:
                self.config_errors.append(
                    f"Key {key} in section {section} is non-positive, "
                    f"but it most be positive.\n"
                )

    def _check_less_than_one_attribute(
        self, section: str, key: str, key_attributes: list[str]
    ) -> None:
        """Checks, whether key is less than one.

        Args:
            section: The corresponding section
            key: The corresponding key
            key_attributes: The list of attributes for key.

        """
        if "less_than_one" in key_attributes:
            if self.getfloat(section, key) >= 1:
                self.config_errors.append(
                    f"Key {key} in section {section} is larger than one, "
                    f"but it must be smaller.\n"
                )

    def _check_larger_than_one_attribute(
        self, section: str, key: str, key_attributes: list[str]
    ) -> None:
        """Checks, whether key is larger than one.

        Args:
            section: The corresponding section
            key: The corresponding key
            key_attributes: The list of attributes for key.

        """
        if "larger_than_one" in key_attributes:
            if self.getfloat(section, key) <= 1:
                self.config_errors.append(
                    f"Key {key} in section {section} is smaller than one, "
                    f"but it must be larger.\n"
                )
