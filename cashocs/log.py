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

"""Logging for cashocs."""

from __future__ import annotations

import datetime
import functools
import logging
from typing import Any, Callable, TYPE_CHECKING, TypeVar

import fenics

from cashocs import mpi

if TYPE_CHECKING:
    from mpi4py import MPI


class LogLevel:
    """Stores the various log levels of cashocs."""

    TRACE = logging.DEBUG - 5
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


TRACE = logging.DEBUG - 5
DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

logging.addLevelName(TRACE, "TRACE")


class Logger:
    """Base class for logging."""

    def __init__(self, name: str) -> None:
        """Initializes the logger.

        Args:
            name (str): The name of the logger.

        """
        h = logging.StreamHandler()
        h.setLevel(logging.INFO)
        self._handler = h

        self._log = logging.getLogger(name)
        self._log.addHandler(h)
        self._log.setLevel(TRACE)

        self._logfiles: dict[str, logging.FileHandler] = {}
        self._indent_level = 0
        self._use_timestamp = True

        self._time_stack: list[datetime.datetime] = []
        self._group_stack: list[str] = []
        self._level_stack: list[int] = []

        self.comm = mpi.COMM_WORLD

    def set_comm(self, comm: MPI.Comm) -> None:
        """Sets the MPI communicator used for logging.

        This should be the same that is supplied for the mesh generation, otherwise
        the program could hang indefinitely.

        Args:
            comm (MPI.Comm): The MPI communicator for logging.

        """
        self.comm = comm

    def log(self, level: int, message: str) -> None:
        """Use the logging functionality of the logger to log to (various) handlers.

        Args:
            level (int): The log level of the message, same as the ones used in the
                python logging module.
            message (str): The message that should be logged.

        """
        if self.comm.rank == 0:
            self._log.log(level, self._format(message))
        self.comm.barrier()

    def trace(self, message: str) -> None:
        """Issues a message at the trace level.

        Args:
            message (str): The message that should be logged.

        """
        self.log(TRACE, message)

    def debug(self, message: str) -> None:
        """Issues a message at the debug level.

        Args:
            message (str): The message that should be logged.

        """
        self.log(logging.DEBUG, message)

    def info(self, message: str) -> None:
        """Issues a message at the info level.

        Args:
            message (str): The message that should be logged.

        """
        self.log(logging.INFO, message)

    def warning(self, message: str) -> None:
        """Issues a message at the warning level.

        Args:
            message (str): The message that should be logged.

        """
        self.log(logging.WARNING, message)

    def error(self, message: str) -> None:
        """Issues a message at the error level.

        Note that this does not raise an exception at the moment.

        Args:
            message (str): The message that should be logged.

        """
        self.log(logging.ERROR, message)

    def critical(self, message: str) -> None:
        """Issues a message at the critical level.

        Note that this does not raise an exception at the moment.

        Args:
            message (str): The message that should be logged.

        """
        self.log(logging.CRITICAL, message)

    def begin(self, message: str, level: int = logging.INFO) -> None:
        """This signals the beginning of a (timed) block of logs.

        This is closed with a suitable call to :py:func:`cashocs.log.end` with which
        each call to :py:func:`cashocs.log.begin` has to be accompanied by.

        Args:
            message (str): The message indicating what block is started.
            level (int, optional): The log level used for issuing the messages.
                Defaults to logging.INFO.

        """
        self._push_message(message)
        self._push_level(level)
        self._push_time()

        start_message = "Start: " + message
        self.log(level, start_message)
        self.log(level, "-" * len(start_message))
        self._add_indent()

    def end(self) -> None:
        """This signals the end of a block started with :py:func:`cashocs.log.begin`."""
        elapsed_time = self._pop_time()
        message = self._pop_message()
        level = self._pop_level()
        self._add_indent(-1)

        end_message = "Finish: " + message + f" -- Elapsed time: {elapsed_time}\n"
        self.log(level, end_message)

    def _add_indent(self, increment: int = 1) -> None:
        """This method adds an indent to the log.

        Args:
            increment (int, optional): The amount of indenting to do. Typically is +1
                or -1. Defaults to 1.

        """
        self._indent_level += increment

    def set_log_level(self, level: int) -> None:
        """This method sets the log level of the default handler, i.e., the console.

        Args:
            level (int): The log level that should be used for the default handler.

        """
        self._handler.setLevel(level)

    def _push_time(self) -> None:
        """Pushes the current time to the time stack."""
        self._time_stack.append(datetime.datetime.now())

    def _pop_time(self) -> datetime.timedelta:
        """Generate a timedelta between the start and end time.

        Returns:
            The timedelta between the :py:func:`cashocs.log.begin` and
            :py:func:`cashocs.log.end` calls.

        """
        start_time = self._time_stack.pop()
        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time

        return elapsed_time

    def _push_message(self, message: str) -> None:
        """Pushes the message to the message stack.

        Args:
            message (str): The message that should be pushed to the stack.

        """
        self._group_stack.append(message)

    def _pop_message(self) -> str:
        """Retrieves a message from the message stack.

        Returns:
            The message that was on the top of the message stack.

        """
        return self._group_stack.pop()

    def _push_level(self, level: int) -> None:
        """Pushes the log level to the level stack.

        Args:
            level (int): The log level.

        """
        self._level_stack.append(level)

    def _pop_level(self) -> int:
        """Retrieves the log level from the level stack.

        Returns:
            The log level that was on top of the level stack.

        """
        return self._level_stack.pop()

    def _format(self, message: str) -> str:
        """Formats a message based on indent and time stamps for logging.

        Args:
            message (str): The input string which should be formatted.

        Returns:
            The formatted string.

        """
        if self._use_timestamp:
            timestamp = datetime.datetime.now().isoformat()
            timestamp = timestamp + " | "
        else:
            timestamp = ""

        indent = 2 * self._indent_level * " "
        return "\n".join([timestamp + indent + line for line in message.split("\n")])

    def add_logfile(
        self, filename: str, mode: str = "a", level: int = logging.DEBUG
    ) -> logging.FileHandler:
        """Adds a file handler to the logger.

        Args:
            filename (str): The path to the file which is used for logging.
            mode (str, optional): The mode with which the log file should be treated.
                "a" appends to the file and "w" overwrites the file. Defaults to "a".
            level (int, optional): The log level used for logging to the file.
                Defaults to logging.DEBUG.

        Returns:
            The file handler for the log file.

        """
        if filename in self._logfiles:
            self.warning(f"Adding logfile {filename} multiple times.")
            return self._logfiles[filename]
        h = logging.FileHandler(filename, mode)
        h.setLevel(level)
        self._log.addHandler(h)
        self._logfiles[filename] = h
        return h

    def add_handler(self, handler: logging.Handler) -> None:
        """Adds an additional handler to the logger.

        Args:
            handler (logging.Handler): The handler that should be added to the logger.

        """
        self._log.addHandler(handler)

    def add_timestamps(self) -> None:
        """This function adds a time stamp to the logged events."""
        self._use_timestamp = True

    def remove_timestamps(self) -> None:
        """This method removes the time stamp from the logged events."""
        self._use_timestamp = False


cashocs_logger = Logger("cashocs")

trace = cashocs_logger.trace
debug = cashocs_logger.debug
info = cashocs_logger.info
warning = cashocs_logger.warning
error = cashocs_logger.error
critical = cashocs_logger.critical

begin = cashocs_logger.begin
end = cashocs_logger.end

set_log_level = cashocs_logger.set_log_level
add_logfile = cashocs_logger.add_logfile
add_timestamps = cashocs_logger.add_timestamps
remove_timestamps = cashocs_logger.remove_timestamps
add_handler = cashocs_logger.add_handler
set_comm = cashocs_logger.set_comm

fenics.set_log_level(fenics.LogLevel.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
logging.getLogger("FFC").setLevel(logging.WARNING)


T = TypeVar("T")


def profile_execution_time(
    action: str, level: int = TRACE
) -> Callable[[Callable[..., T]], Callable[..., T]]:
    """Profiles the execution time of a function.

    This decorator is used to determine how long it takes to complete a function call.
    The output consists of log calls specified by the log level.

    Args:
        action (str): A string describing what the function does.
        level (int): The log level for the output of the profiling.

    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            start_time = datetime.datetime.now()

            result = func(*args, **kwargs)

            end_time = datetime.datetime.now()
            elapsed_time = end_time - start_time
            cashocs_logger.log(level, f"Elapsed time for {action}: {elapsed_time}.\n")
            return result

        return wrapper

    return decorator
