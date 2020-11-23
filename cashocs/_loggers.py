# Copyright (C) 2020 Sebastian Blauth
#
# This file is part of CASHOCS.
#
# CASHOCS is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# CASHOCS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with CASHOCS.  If not, see <https://www.gnu.org/licenses/>.

"""Logging module for cashocs.

"""

import logging



class LogLevel:
	"""Stores the various log levels of cashocs.
	
	See Also
	--------
	cashocs.set_log_level : sets the log level of cashocs
	
	"""
	
	DEBUG = logging.DEBUG
	INFO = logging.INFO
	WARNING = logging.WARNING
	ERROR = logging.ERROR
	CRITICAL = logging.CRITICAL
	
	def __init__(self):
		pass



logging.basicConfig(format='%(name)s - %(levelname)s : %(message)s')
_cashocs_logger = logging.getLogger('cashocs')
_cashocs_logger.setLevel(LogLevel.INFO)




def set_log_level(level):
	"""Determines the log level of cashocs.
	
	Can be used to show, e.g., info and warning messages or to hide them.
	There are a total of five different levels for the logs: ``DEBUG``, ``INFO``,
	``WARNING``, ``ERROR``, and ``CRITICAL``. The usage of this method is explained in
	the examples section.
	
	Parameters
	----------
	level : int
		Should be one of ``cashocs.LogLevel.DEBUG``,
		``cashocs.LogLevel.INFO``, ``cashocs.LogLevel.WARNING``,
		``cashocs.LogLevel.ERROR``, ``cashocs.LogLevel.CRITICAL``

	Returns
	-------
	None
	
	Notes
	-----
	The log level setting is global, so if you use this interactively,
	you have to restart / reload your interactive console to return to
	the default settings.
	
	Examples
	--------
	To set the log level of cashocs, use this method as follows::
	
		import cashocs
		
		cashocs.set_log_level(cashocs.LogLevel.WARNING)

	which only shows messages with a level of ``WARNING`` or higher.
	To use a different level, replace ``WARNING`` by ``DEBUG``, ``INFO``, ``ERROR``,
	or ``CRITICAL``.
	
	"""
	
	_cashocs_logger.setLevel(level)



def debug(message):
	"""Issues a debug level logging message.
	
	Parameters
	----------
	message : str
		The message to be issued.

	Returns
	-------
	None
	"""
	
	_cashocs_logger.debug(message)



def info(message):
	"""Issues an info level logging message.
	
	Parameters
	----------
	message : str
		The message to be issued.

	Returns
	-------
	None
	"""
	
	_cashocs_logger.info(message)
	
	
	
def warning(message):
	"""Issues a warning level logging message.
	
	Parameters
	----------
	message : str
		The message to be issued.

	Returns
	-------
	None
	"""
	
	_cashocs_logger.warning(message)



def error(message):
	"""Issues a error level logging message.
	
	Parameters
	----------
	message : str
		The message to be issued.

	Returns
	-------
	None
	"""
	
	_cashocs_logger.error(message, exc_info=True)



def critical(message):
	"""Issues a critical level logging message.
	
	Parameters
	----------
	message : str
		The message to be issued.

	Returns
	-------
	None
	"""
	
	_cashocs_logger.critical(message, exc_info=True)
