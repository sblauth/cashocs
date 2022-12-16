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

"""Logging for cashocs."""

from __future__ import annotations

import logging
from typing import Any

import fenics


class CashocsFormatter(logging.Formatter):
    """Logging Formatter for colored output."""

    my_format = "%(name)s - %(levelname)s - %(message)s"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """See base class."""
        super().__init__(*args, **kwargs)

    def format(self, record: logging.LogRecord) -> str:
        """See base class."""
        formatter = logging.Formatter(self.my_format)
        return formatter.format(record)


class LogLevel:
    """Stores the various log levels of cashocs."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


_cashocs_logger = logging.getLogger("cashocs")
_cashocs_handler = logging.StreamHandler()
_cashocs_formatter = CashocsFormatter()
_cashocs_handler.setFormatter(_cashocs_formatter)
_cashocs_logger.addHandler(_cashocs_handler)
_cashocs_logger.setLevel(LogLevel.INFO)

fenics.set_log_level(fenics.LogLevel.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
logging.getLogger("FFC").setLevel(logging.WARNING)


def set_log_level(level: int) -> None:
    """Determines the log level of cashocs.

    Can be used to show, e.g., info and warning messages or to hide them. There are a
    total of five different levels for the logs: ``DEBUG``, ``INFO``, ``WARNING``,
    ``ERROR``, and ``CRITICAL``. The usage of this method is explained in the examples
    section.

    Args:
        level: Should be one of ``cashocs.LogLevel.DEBUG``, ``cashocs.LogLevel.INFO``,
            ``cashocs.LogLevel.WARNING``, ``cashocs.LogLevel.ERROR``,
            ``cashocs.LogLevel.CRITICAL``

    Notes:
        The log level setting is global, so if you use this interactively, you have to
        restart / reload your interactive console to return to the default settings.

    Examples:
        To set the log level of cashocs, use this method as follows::

            import cashocs

            cashocs.set_log_level(cashocs.LogLevel.WARNING)

        which only shows messages with a level of ``WARNING`` or higher.
        To use a different level, replace ``WARNING`` by ``DEBUG``, ``INFO``, ``ERROR``,
        or ``CRITICAL``.

    """
    _cashocs_logger.setLevel(level)


def debug(message: str) -> None:
    """Issues a debug level logging message.

    Args:
        message: The message to be issued.

    """
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        _cashocs_logger.debug(message)
    fenics.MPI.barrier(fenics.MPI.comm_world)


def info(message: str) -> None:
    """Issues an info level logging message.

    Args:
        message: The message to be issued.

    """
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        _cashocs_logger.info(message)
    fenics.MPI.barrier(fenics.MPI.comm_world)


def warning(message: str) -> None:
    """Issues a warning level logging message.

    Args:
        message: The message to be issued.

    """
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        _cashocs_logger.warning(message)
    fenics.MPI.barrier(fenics.MPI.comm_world)


def error(message: str) -> None:
    """Issues a error level logging message.

    Args:
        message: The message to be issued.

    """
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        _cashocs_logger.error(message)
    fenics.MPI.barrier(fenics.MPI.comm_world)


def critical(message: str) -> None:
    """Issues a critical level logging message.

    Args:
        message: The message to be issued.

    """
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        _cashocs_logger.critical(message)
    fenics.MPI.barrier(fenics.MPI.comm_world)
