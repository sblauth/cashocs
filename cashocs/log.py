# Copyright (C) 2020-2024 Fraunhofer ITWM and Sebastian Blauth
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
import logging

import fenics


class Logger:
    """Base class for logging."""

    def __init__(self, name: str) -> None:
        """Initializes the logger.

        Args:
            name (str): The name of the logger.

        """
        h = logging.StreamHandler()
        h.setLevel(logging.WARNING)
        self._handler = h

        self._log = logging.getLogger(name)
        self._log.addHandler(h)
        self._log.setLevel(logging.DEBUG)

        self._logfiles: dict[str, logging.FileHandler] = {}
        self._indent_level = 0
        self._use_timestamp = False

        self._time_stack: list[datetime.datetime] = []
        self._group_stack: list[str] = []
        self._level_stack: list[int] = []

    def log(self, level: int, message: str) -> None:
        """Use the logging functionality of the logger to log to (various) handlers.

        Args:
            level (int): The log level of the message, same as the ones used in the
                python logging module.
            message (str): The message that should be logged.

        """
        if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
            self._log.log(level, self._format(message))
        fenics.MPI.barrier(fenics.MPI.comm_world)

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

        This is closed with a suitable call to :py:`log.end` with which each call to
        :py:`log.begin` has to be accompanied by.

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
        """This signals the end of a block started with :py:`log.begin`."""
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
            datetime.timedelta: The timedelta between the :py:`log.begin` and
            :py:`log.end` calls.

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
            str: The message that was on the top of the message stack.

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
            int: The log level that was on top of the level stack.

        """
        return self._level_stack.pop()

    def _format(self, message: str) -> str:
        """Formats a message based on indent and time stamps for logging.

        Args:
            message (str): The input string which should be formatted.

        Returns:
            str: The formatted string.

        """
        if self._use_timestamp:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            timestamp = timestamp + " | "
        else:
            timestamp = ""

        indent = 2 * self._indent_level * " "
        return "\n".join([indent + timestamp + line for line in message.split("\n")])

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
            logging.FileHandler: The file handler for the log file.

        """
        if filename in self._logfiles:
            self.warning(f"Adding logfile {filename} multiple times.")
            return self._logfiles[filename]
        h = logging.FileHandler(filename, mode)
        h.setLevel(level)
        self._log.addHandler(h)
        self._logfiles[filename] = h
        return h

    def add_timestamp(self) -> None:
        """This function adds a time stamp to the logged events."""
        self._use_timestamp = True

    def remove_timestamp(self) -> None:
        """This method removes the time stamp from the logged events."""
        self._use_timestamp = False


cashocs_logger = Logger("cashocs")

debug = cashocs_logger.debug
info = cashocs_logger.info
warning = cashocs_logger.warning
error = cashocs_logger.error
critical = cashocs_logger.critical

begin = cashocs_logger.begin
end = cashocs_logger.end

set_log_level = cashocs_logger.set_log_level
add_logfile = cashocs_logger.add_logfile
add_timestamp = cashocs_logger.add_timestamp
remove_timestamp = cashocs_logger.remove_timestamp


class LogLevel:
    """Stores the various log levels of cashocs."""

    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL

fenics.set_log_level(fenics.LogLevel.WARNING)
logging.getLogger("UFL").setLevel(logging.WARNING)
logging.getLogger("FFC").setLevel(logging.WARNING)
