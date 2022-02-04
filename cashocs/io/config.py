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

"""Module for managing config files."""

from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import Optional, List

from cashocs import _exceptions
from cashocs import _loggers


# deprecated
def create_config(path: str) -> ConfigParser:  # pragma: no cover
    """Loads a config object from a config file.

    Loads the config from a .ini file via the configparser package.

    Args:
        path: The path to the .ini file storing the configuration.

    Returns:
        The output config file, which includes the path to the .ini file.

    .. deprecated:: 1.1.0
        This is replaced by :py:func:`load_config <cashocs.load_config>`
        and will be removed in the future.
    """

    _loggers.warning(
        "DEPRECATION WARNING: cashocs.create_config is replaced by cashocs.load_config "
        "and will be removed in the future."
    )
    config = load_config(path)

    return config


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
        if not (char.isdigit() or char.isspace() or char in ["[", "]", ".", ",", "-"]):
            return result

    if string[0] != "[":
        return result
    if string[-1] != "]":
        return result

    result = True

    return result


class Config(ConfigParser):
    """Class for handling the config in cashocs."""

    def __init__(self, config_file: Optional[str] = None) -> None:
        """
        Args:
            config_file: Path to the config file.
        """

        super().__init__()

        self.config_scheme = {
            "Mesh": {
                "mesh_file": {
                    "type": "str",
                    "attributes": ["file"],
                    "file_extension": "xdmf",
                },
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
                    "fallback": False,
                    "requires": [("Mesh", "gmsh_file")],
                },
                "show_gmsh_output": {"type": "bool", "fallback": False},
            },
            "StateSystem": {
                "is_linear": {"type": "bool", "fallback": False},
                "newton_rtol": {
                    "type": "float",
                    "fallback": 1e-11,
                    "attributes": ["less_than_one", "positive"],
                },
                "newton_atol": {
                    "type": "float",
                    "fallback": 1e-13,
                    "attributes": ["non_negative"],
                },
                "newton_iter": {
                    "type": "int",
                    "fallback": 50,
                    "attributes": ["non_negative"],
                },
                "newton_damped": {"type": "bool", "fallback": True},
                "newton_inexact": {"type": "bool", "fallback": False},
                "newton_verbose": {"type": "bool", "fallback": False},
                "picard_iteration": {"type": "bool", "fallback": False},
                "picard_rtol": {
                    "type": "float",
                    "fallback": 1e-10,
                    "attributes": ["positive", "less_than_one"],
                },
                "picard_atol": {
                    "type": "float",
                    "fallback": 1e-20,
                    "attributes": ["non_negative"],
                },
                "picard_iter": {
                    "type": "int",
                    "fallback": 50,
                    "attributes": ["non_negative"],
                },
                "picard_verbose": {"type": "bool", "fallback": False},
            },
            "OptimizationRoutine": {
                "algorithm": {
                    "type": "str",
                    "fallback": "none",
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
                        "none",
                    ],
                },
                "rtol": {
                    "type": "float",
                    "fallback": 1e-3,
                    "attributes": ["less_than_one", "positive"],
                },
                "atol": {
                    "type": "float",
                    "fallback": 0.0,
                    "attributes": ["non_negative"],
                },
                "maximum_iterations": {
                    "type": "int",
                    "fallback": 100,
                    "attributes": ["non_negative"],
                },
                "initial_stepsize": {
                    "type": "float",
                    "fallback": 1.0,
                    "attributes": ["positive"],
                },
                "epsilon_armijo": {
                    "type": "float",
                    "fallback": 1e-4,
                    "attributes": ["positive", "less_than_one"],
                },
                "beta_armijo": {
                    "type": "float",
                    "fallback": 2.0,
                    "attributes": ["positive", "larger_than_one"],
                },
                "gradient_method": {
                    "type": "str",
                    "fallback": "direct",
                    "possible_options": ["direct", "iterative"],
                },
                "gradient_tol": {
                    "type": "float",
                    "fallback": 1e-9,
                    "attributes": ["less_than_one", "positive"],
                },
                "soft_exit": {"type": "bool", "fallback": False},
            },
            "AlgoLBFGS": {
                "bfgs_memory_size": {
                    "type": "int",
                    "fallback": 5,
                    "attributes": ["non_negative"],
                },
                "use_bfgs_scaling": {"type": "bool", "fallback": True},
            },
            "AlgoCG": {
                "cg_method": {
                    "type": "str",
                    "fallback": "fr",
                    "possible_options": ["fr", "pr", "hs", "dy", "hz"],
                },
                "cg_periodic_restart": {"type": "bool", "fallback": False},
                "cg_periodic_its": {
                    "type": "int",
                    "fallback": 10,
                    "attributes": ["non_negative"],
                },
                "cg_relative_restart": {"type": "bool", "fallback": False},
                "cg_restart_tol": {
                    "type": "float",
                    "fallback": 0.25,
                    "attributes": ["positive"],
                },
            },
            "AlgoTNM": {
                "inner_newton": {
                    "type": "str",
                    "fallback": "cr",
                    "possible_options": ["cg", "cr"],
                },
                "inner_newton_rtol": {
                    "type": "float",
                    "fallback": 1e-15,
                    "attributes": ["positive", "less_than_one"],
                },
                "inner_newton_atol": {
                    "type": "float",
                    "fallback": 0.0,
                    "attributes": ["non_negative"],
                },
                "max_it_inner_newton": {
                    "type": "int",
                    "fallback": 50,
                    "attributes": ["non_negative"],
                },
            },
            "ShapeGradient": {
                "shape_bdry_def": {"type": "list", "fallback": "[]"},
                "shape_bdry_fix": {"type": "list", "fallback": "[]"},
                "shape_bdry_fix_x": {"type": "list", "fallback": "[]"},
                "shape_bdry_fix_y": {"type": "list", "fallback": "[]"},
                "shape_bdry_fix_z": {"type": "list", "fallback": "[]"},
                "use_pull_back": {"type": "bool", "fallback": True},
                "lambda_lame": {"type": "float", "fallback": 0.0},
                "damping_factor": {
                    "type": "float",
                    "fallback": 0.0,
                    "attributes": ["non_negative"],
                },
                "mu_def": {
                    "type": "float",
                    "fallback": 1.0,
                    "attributes": ["positive"],
                },
                "mu_fix": {
                    "type": "float",
                    "fallback": 1.0,
                    "attributes": ["positive"],
                },
                "use_sqrt_mu": {"type": "bool", "fallback": False},
                "inhomogeneous": {"type": "bool", "fallback": False},
                "update_inhomogeneous": {"type": "bool", "fallback": False},
                "use_distance_mu": {"type": "bool", "fallback": False},
                "dist_min": {
                    "type": "float",
                    "fallback": 1.0,
                    "attributes": ["non_negative"],
                },
                "dist_max": {
                    "type": "float",
                    "fallback": 1.0,
                    "larger_equal_than": ("ShapeGradient", "dist_min"),
                    "attributes": ["non_negative"],
                },
                "mu_min": {
                    "type": "float",
                    "fallback": 1.0,
                    "attributes": ["positive"],
                },
                "mu_max": {
                    "type": "float",
                    "fallback": 1.0,
                    "attributes": ["positive"],
                },
                "boundaries_dist": {"type": "list", "fallback": "[]"},
                "smooth_mu": {"type": "bool", "fallback": False},
                "use_p_laplacian": {"type": "bool", "fallback": False},
                "p_laplacian_power": {
                    "type": "int",
                    "fallback": 2,
                    "attributes": ["larger_than_one"],
                },
                "p_laplacian_stabilization": {
                    "type": "float",
                    "fallback": 0.0,
                    "attributes": ["non_negative", "less_than_one"],
                },
                "fixed_dimensions": {
                    "type": "list",
                    "fallback": "[]",
                    "conflicts": [("ShapeGradient", "use_p_laplacian")],
                },
                "degree_estimation": {"type": "bool", "fallback": True},
            },
            "Regularization": {
                "factor_volume": {
                    "type": "float",
                    "fallback": 0.0,
                    "attributes": ["non_negative"],
                },
                "target_volume": {
                    "type": "float",
                    "fallback": 0.0,
                    "attributes": ["non_negative"],
                },
                "use_initial_volume": {"type": "bool", "fallback": False},
                "factor_surface": {
                    "type": "float",
                    "fallback": 0.0,
                    "attributes": ["non_negative"],
                },
                "target_surface": {
                    "type": "float",
                    "fallback": 0.0,
                    "attributes": ["non_negative"],
                },
                "use_initial_surface": {"type": "bool", "fallback": False},
                "factor_curvature": {
                    "type": "float",
                    "fallback": 0.0,
                    "attributes": ["non_negative"],
                },
                "factor_barycenter": {
                    "type": "float",
                    "fallback": 0.0,
                    "attributes": ["non_negative"],
                },
                "target_barycenter": {"type": "list", "fallback": "[0,0,0]"},
                "use_initial_barycenter": {"type": "bool", "fallback": False},
                "measure_hole": {"type": "bool", "fallback": False},
                "x_start": {"type": "float", "fallback": 0.0},
                "x_end": {
                    "type": "float",
                    "fallback": 1.0,
                    "larger_than": ("Regularization", "x_start"),
                },
                "y_start": {"type": "float", "fallback": 0.0},
                "y_end": {
                    "type": "float",
                    "fallback": 1.0,
                    "larger_than": ("Regularization", "y_start"),
                },
                "z_start": {"type": "float", "fallback": 0.0},
                "z_end": {
                    "type": "float",
                    "fallback": 1.0,
                    "larger_than": ("Regularization", "z_start"),
                },
                "use_relative_scaling": {"type": "bool", "fallback": False},
            },
            "MeshQuality": {
                "volume_change": {
                    "type": "float",
                    "fallback": float("inf"),
                    "attributes": ["positive", "larger_than_one"],
                },
                "angle_change": {
                    "type": "float",
                    "fallback": float("inf"),
                    "attributes": ["positive"],
                },
                "tol_lower": {
                    "type": "float",
                    "fallback": 0.0,
                    "attributes": ["less_than_one", "non_negative"],
                },
                "tol_upper": {
                    "type": "float",
                    "fallback": 1e-15,
                    "attributes": ["less_than_one", "positive"],
                    "larger_than": ("MeshQuality", "tol_lower"),
                },
                "measure": {
                    "type": "str",
                    "fallback": "skewness",
                    "possible_options": [
                        "skewness",
                        "radius_ratios",
                        "maximum_angle",
                        "condition_number",
                    ],
                },
                "type": {
                    "type": "str",
                    "fallback": "min",
                    "possible_options": ["min", "avg", "minimum", "average"],
                },
            },
            "Output": {
                "verbose": {"type": "bool", "fallback": True},
                "save_results": {"type": "bool", "fallback": True},
                "save_txt": {"type": "bool", "fallback": True},
                "save_pvd": {"type": "bool", "fallback": False},
                "save_pvd_adjoint": {"type": "bool", "fallback": False},
                "save_pvd_gradient": {"type": "bool", "fallback": False},
                "save_mesh": {
                    "type": "bool",
                    "fallback": False,
                    "requires": [("Mesh", "gmsh_file")],
                },
                "result_dir": {"type": "str", "fallback": "./results"},
                "time_suffix": {"type": "bool", "fallback": False},
            },
            "Debug": {
                "remeshing": {"type": "bool", "fallback": False},
                "restart": {"type": "bool", "fallback": False},
            },
            "DEFAULT": {},
        }
        self.config_errors = []

        if config_file is not None:
            self.read(config_file)

    def validate_config(self) -> None:
        """Validates the configuration file."""

        self._check_sections()
        self._check_keys()

        if len(self.config_errors) > 0:
            raise _exceptions.ConfigError(self.config_errors)

    def _check_sections(self) -> None:
        """Checks whether all sections are valid."""

        for section_name, section in self.items():
            if section_name not in self.config_scheme.keys():
                self.config_errors.append(
                    f"The following section is not valid: {section}\n"
                )

    def _check_keys(self) -> None:
        """Checks the keys of the sections."""

        for section_name, section in self.items():
            for key in section.keys():
                if section_name in self.config_scheme.keys():
                    if key not in self.config_scheme[section_name].keys():
                        self.config_errors.append(
                            f"Key {key} is not valid for section {section_name}.\n"
                        )
                    else:
                        self._check_key_type(section_name, key)
                        self._check_possible_options(section_name, key)
                        self._check_attributes(section_name, key)
                        self._check_key_requirements(section_name, key)
                        self._check_key_conflicts(section_name, key)
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

    def _check_key_conflicts(self, section: str, key: str) -> None:
        """Checks, whether conflicting keys are present.

        Args:
            section: The corresponding section
            key: The corresponding key
        """

        if self.has_option(section, key):
            if "conflicts" in self.config_scheme[section][key].keys():
                conflicts = self.config_scheme[section][key]["conflicts"]
                for conflict in conflicts:
                    if self.has_option(conflict[0], conflict[1]):
                        if self.getboolean(conflict[0], conflict[1]):
                            self.config_errors.append(
                                f"Key {conflict[1]} in section {conflict[0]} "
                                f"conflicts with key {key} in section {section}.\n"
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
            if self.has_option(partner[0], partner[1]):
                lower_value = self.getfloat(partner[0], partner[1])
            else:
                lower_value = self.config_scheme[partner[0]][partner[1]]["fallback"]
            if not lower_value < higher_value:
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
            if self.has_option(partner[0], partner[1]):
                lower_value = self.getfloat(partner[0], partner[1])
            else:
                lower_value = self.config_scheme[partner[0]][partner[1]]["fallback"]
            if not lower_value <= higher_value:
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
        self, section: str, key: str, key_attributes: List[str]
    ) -> None:
        """Checks, whether a file specified in key exists.

        Args:
            section: The corresponding section
            key: The corresponding key
            key_attributes: The list of attributes for key.
        """

        if "file" in key_attributes:
            file = Path(self.get(section, key))
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
        self, section: str, key: str, key_attributes: List[str]
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
        self, section: str, key: str, key_attributes: List[str]
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
        self, section: str, key: str, key_attributes: List[str]
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
        self, section: str, key: str, key_attributes: List[str]
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
