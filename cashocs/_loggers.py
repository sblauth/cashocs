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
	"""Determines the log level
	
	Can be used to show, e.g., info and warning messages or to hide them.
	
	Parameters
	----------
	level : int
		Should be one of cashocs.LogLevel.DEBUG,
		cashocs.LogLevel.INFO, cashocs.LogLevel.WARNING,
		cashocs.LogLevel.ERROR, cashocs.LogLevel.CRITICAL

	Returns
	-------
	None

	"""
	
	_cashocs_logger.setLevel(level)



def debug(message):
	_cashocs_logger.debug(message)



def info(message):
	_cashocs_logger.info(message)
	
	
	
def warning(message):
	_cashocs_logger.warning(message)



def error(message):
	_cashocs_logger.error(message, exc_info=True)



def critical(message):
	_cashocs_logger.critical(message, exc_info=True)
