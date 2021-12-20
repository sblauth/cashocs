"""
Created on 20/12/2021, 13.26

@author: blauths
"""

from configparser import ConfigParser
from pathlib import Path
from typing import Optional

from ._exceptions import ConfigError


def _check_for_config_list(string: str) -> bool:
    """Checks, whether a given string is a valid representation of a list of numbers (floats)

    Parameters
    ----------
    string : str
        The input string.

    Returns
    -------
    bool
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
    def __init__(self, config_file: Optional[str] = None):
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
                "remesh": {"type": "bool"},
                "show_gmsh_output": {"type": "bool"},
            },
            "StateSystem": {
                "is_linear": {"type": "bool"},
                "newton_rtol": {
                    "type": "float",
                    "attributes": ["less_than_one", "positive"],
                },
                "newton_atol": {"type": "float", "attributes": ["positive"]},
                "newton_iter": {"type": "int", "attributes": ["non_negative"]},
                "newton_damped": {"type": "bool"},
                "newton_inexact": {"type": "bool"},
                "newton_verbose": {"type": "bool"},
                "picard_iteration": {"type": "bool"},
                "picard_rtol": {
                    "type": "float",
                    "attributes": ["positive", "less_than_one"],
                },
                "picard_atol": {"type": "float", "attributes": ["positive"]},
                "picard_iter": {"type": "int", "attributes": ["non_negative"]},
                "picard_verbose": {"type": "bool"},
            },
            "OptimizationRoutine": {
                "algorithm": {
                    "type": "str",
                    "possible_options": {
                        "gd",
                        "gradient_descent",
                        "bfgs",
                        "lbfgs",
                        "nonlinear_cg",
                        "ncg",
                        "nonlinear_conjugate_gradient",
                        "conjugate_gradient",
                        "newton",
                        "pdas",
                        "primal_dual_active_set",
                    },
                },
                "rtol": {"type": "float", "attributes": ["less_than_one", "positive"]},
                "atol": {"type": "float", "attributes": ["non_negative"]},
                "maximum_iterations": {"type": "int", "attributes": ["non_negative"]},
                "initial_stepsize": {"type": "float", "attributes": ["positive"]},
                "epsilon_armijo": {
                    "type": "float",
                    "attributes": ["positive", "less_than_one"],
                },
                "beta_armijo": {
                    "type": "float",
                    "attributes": ["positive", "larger_than_one"],
                },
                "gradient_method": {
                    "type": "str",
                    "possible_options": {"direct", "iterative"},
                },
                "gradient_tol": {
                    "type": "float",
                    "attributes": ["less_than_one", "positive"],
                },
                "soft_exit": {"type": "bool"},
            },
            "AlgoLBFGS": {
                "bfgs_memory_size": {"type": "int", "attributes": ["non_negative"]},
                "use_bfgs_scaling": {"type": "bool"},
            },
            "AlgoCG": {
                "cg_method": {
                    "type": "str",
                    "possible_options": {"FR", "PR", "HS", "DY", "HZ"},
                },
                "cg_periodic_restart": {"type": "bool"},
                "cg_periodic_its": {"type": "int", "attributes": ["non_negative"]},
                "cg_relative_restart": {"type": "bool"},
                "cg_restart_tol": {
                    "type": "float",
                    "attributes": ["positive"],
                },
            },
            "AlgoTNM": {
                "inner_newton": {"type": "str", "possible_options": {"cg", "cr"}},
                "inner_newton_tolerance": {
                    "type": "float",
                    "attributes": ["positive", "less_than_one"],
                },
                "max_it_inner_newton": {"type": "int", "attributes": ["non_negative"]},
            },
            "AlgoPDAS": {
                "inner_pdas": {
                    "type": "str",
                    "possible_options": {
                        "gd",
                        "gradient_descent",
                        "bfgs",
                        "lbfgs",
                        "nonlinear_cg",
                        "ncg",
                        "cg",
                        "nonlinear_conjugate_gradient",
                        "conjugate_gradient",
                        "newton",
                    },
                },
                "pdas_inner_tolerance": {
                    "type": "float",
                    "attributes": ["positive", "less_than_one"],
                },
                "maximum_iterations_inner_pdas": {
                    "type": "int",
                    "attributes": ["non_negative"],
                },
                "pdas_regularization_parameter": {
                    "type": "float",
                    "attributes": ["non_negative"],
                },
            },
            "ShapeGradient": {
                "shape_bdry_def": {"type": "list"},
                "shape_bdry_fix": {"type": "list"},
                "shape_bdry_fix_x": {"type": "list"},
                "shape_bdry_fix_y": {"type": "list"},
                "shape_bdry_fix_z": {"type": "list"},
                "use_pull_back": {"type": "bool"},
                "lambda_lame": {"type": "float"},
                "damping_factor": {"type": "float", "attributes": ["non_negative"]},
                "mu_def": {"type": "float", "attributes": ["positive"]},
                "mu_fix": {"type": "float", "attributes": ["positive"]},
                "use_sqrt_mu": {"type": "bool"},
                "inhomogeneous": {"type": "bool"},
                "update_inhomogeneous": {"type", "bool"},
                "use_distance_mu": {"type": "bool"},
                "dist_min": {"type": "float", "attributes": ["non_negative"]},
                "dist_max": {"type": "float", "attributes": ["non_negative"]},
                "mu_min": {"type": "float", "attributes": ["positive"]},
                "mu_max": {"type": "float", "attributes": ["positive"]},
                "boundaries_dist": {"type": "list"},
                "smooth_mu": {"type": "bool"},
                "use_p_laplacian": {"type": "bool"},
                "p_laplacian_power": {"type": "int", "attributes": ["larger_than_one"]},
                "p_laplacian_stabilization": {
                    "type": "float",
                    "attributes": ["non_negative", "less_than_one"],
                },
                "fixed_dimensions": {"type": "list"},
                "degree_estimation": {"type": "bool"},
            },
            "Regularization": {
                "factor_volume": {"type": "float", "attributes": ["non_negative"]},
                "target_volume": {"type": "float", "attributes": ["non_negative"]},
                "use_initial_volume": {"type": "bool"},
                "factor_surface": {"type": "float", "attributes": ["non_negative"]},
                "target_surface": {"type": "float", "attributes": ["non_negative"]},
                "use_initial_surface": {"type": "bool"},
                "factor_curvature": {"type": "float", "attributes": ["non_negative"]},
                "factor_barycenter": {"type": "float", "attributes": ["non_negative"]},
                "target_barycenter": {"type": "list"},
                "use_initial_barycenter": {"type": "bool"},
                "measure_hole": {"type": "bool"},
                "use_relative_scaling": {"type": "bool"},
            },
            "MeshQuality": {
                "volume_change": {
                    "type": "float",
                    "attributes": ["positive", "larger_than_one"],
                },
                "angle_change": {"type": "float", "attributes": ["positive"]},
                "tol_lower": {
                    "type": "float",
                    "attributes": ["less_than_one", "non_negative"],
                },
                "tol_upper": {
                    "type": "float",
                    "attributes": ["less_than_one", "non_negative"],
                },
                "measure": {
                    "type": "str",
                    "possible_options": {
                        "skewness",
                        "radius_ratios",
                        "maximum_angle",
                        "condition_number",
                    },
                },
                "type": {
                    "type": "str",
                    "possible_options": {"min", "avg", "minimum", "average"},
                },
                "check_a_posteriori": {"type": "bool"},
            },
            "Output": {
                "verbose": {"type": "bool"},
                "save_results": {"type": "bool"},
                "save_txt": {"type": "bool"},
                "save_pvd": {"type": "bool"},
                "save_pvd_adjoint": {"type": "bool"},
                "save_pvd_gradient": {"type": "bool"},
                "save_mesh": {"type": "bool"},
                "result_dir": {"type": "str"},
                "time_suffix": {"type": "bool"},
            },
            "Debug": {"remeshing": {"type": "bool"}},
            "DEFAULT": {},
        }
        self.config_errors = []

        if config_file is not None:
            self.read(config_file)

    def validate_config(self):
        self._check_sections()
        self._check_keys()

        if len(self.config_errors) > 0:
            raise ConfigError(self.config_errors)

    def _check_sections(self):
        for section_name, section in self.items():
            if section_name not in self.config_scheme.keys():
                self.config_errors.append(
                    f"The following section is not valid: {section}\n"
                )

    def _check_keys(self):
        for section_name, section in self.items():
            for key in section.keys():
                if key not in self.config_scheme[section_name].keys():
                    self.config_errors.append(
                        f"Key {key} is not valid for section {section_name}.\n"
                    )
                else:
                    self._check_key_type(section_name, key)
                    self._check_possible_options(section_name, key)
                    self._check_attributes(section_name, key)

    def _check_key_type(self, section, key):
        key_type = self.config_scheme[section][key]["type"]
        try:
            if key_type == "str":
                self.get(section, key)
            elif key_type == "bool":
                self.getboolean(section, key)
            elif key_type == "int":
                self.getint(section, key)
            elif key_type == "float":
                self.getfloat(section, key)
            elif key_type == "list":
                if not _check_for_config_list(self.get(section, key)):
                    raise ValueError
        except ValueError:
            self.config_errors.append(
                f"Key {key} in section {section} has the wrong type. Required type is {key_type}.\n"
            )

    def _check_possible_options(self, section, key):
        if "possible_options" in self.config_scheme[section][key].keys():
            if (
                self[section][key]
                not in self.config_scheme[section][key]["possible_options"]
            ):
                self.config_errors.append(
                    f"Key {key} in section {section} has a wrong value. Possible options are {self.config_scheme[section][key]['possible_options']}.\n"
                )

    def _check_attributes(self, section, key):
        if "attributes" in self.config_scheme[section][key].keys():
            key_attributes = self.config_scheme[section][key]["attributes"]
            self._check_file_attribute(section, key, key_attributes)
            self._check_non_negative_attribute(section, key, key_attributes)
            self._check_positive_attribute(section, key, key_attributes)
            self._check_less_than_one_attribute(section, key, key_attributes)

    def _check_file_attribute(self, section, key, key_attributes):
        if "file" in key_attributes:
            file = Path(self.get(section, key))
            if not file.is_file():
                self.config_errors.append(
                    f"Key {key} in section {section} should point to a file, but the file does not exist.\n"
                )

            self._check_file_extension(
                section, key, self.config_scheme[section][key]["file_extension"]
            )

    def _check_file_extension(self, section, key, extension):
        path_to_file = self.get(section, key)
        if not path_to_file.split(".")[-1] == extension:
            self.config_errors.append(
                f"Key {key} in section {section} has the wrong file extension, it should end in {extension}.\n"
            )

    def _check_non_negative_attribute(self, section, key, key_attributes):
        if "non_negative" in key_attributes:
            if self.getfloat(section, key) < 0:
                self.config_errors.append(
                    f"Key {key} in section {section} is negative, but it must not be.\n"
                )

    def _check_positive_attribute(self, section, key, key_attributes):
        if "positive" in key_attributes:
            if self.getfloat(section, key) <= 0:
                self.config_errors.append(
                    f"Key {key} in section {section} is non-positive, but it most be positive.\n"
                )

    def _check_less_than_one_attribute(self, section, key, key_attributes):
        if "less_than_one" in key_attributes:
            if self.getfloat(section, key) >= 1:
                self.config_errors.append(
                    f"Key {key} in section {section} is larger than one, but it must be smaller.\n"
                )

    def _check_larger_than_one_attribute(self, section, key, key_attributes):
        if "larger_than_one" in key_attributes:
            if self.getfloat(section, key) <= 1:
                self.config_errors.append(
                    f"Key {key} in section {section} is smaller than one, but it must be larger.\n"
                )
