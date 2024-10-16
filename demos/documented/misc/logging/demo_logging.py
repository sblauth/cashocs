# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

# ```{eval-rst}
# .. include:: ../../../global.rst
# ```
#
# (demo_logging)=
# # Logging with cashocs
#
# ## Topic
#
# In this demo, we investigate how cashocs logging behavior can be configured. Cashocs
# uses the python built-in logging library for logging and has the level `INFO` as
# default log level for its terminal output. This ensures that users can easily see the
# most important parts down to the solution of the PDEs readily in the terminal when
# running script. However, for some users, this might be too cluttered, while others
# would like to have additional information, particularly for debugging purposes.
# These objectives can be readily achieved and in the following, we will discuss how.
#
# ## Implementation
#
# The complete python code can be found in the file {download}`demo_logging.py
# </../../demos/documented/misc/logging/demo_logging.py>`.
#
#
# ## Setting the log level for terminal output
#
# As briefly touched upon in {ref}`demo_poisson`, the log level for cashocs can be set
# with the {py:func}`cashocs.log.set_log_level` as follows

# +
import cashocs

cashocs.log.set_log_level(cashocs.log.INFO)

# -

# As stated previously, there are five log levels available, namely
# {py:class}`cashocs.log.DEBUG`, {py:class}`cashocs.log.INFO`,
# {py:class}`cashocs.log.WARNING`, {py:class}`cashocs.log.ERROR`, and
# {py:class}`cashocs.log.CRITICAL`. The default log level for the terminal output
# is {py:class}`cashocs.log.INFO`, which will print information, which is relevant for
# most users, such as the progress of the optimization or details regarding the solution
# of the (nonlinear) state equation. If this is too much information for you, then you
# can restrict the terminal logs to warnings and higher with a call to

cashocs.log.set_log_level(cashocs.log.WARNING)

# which will print way less details. On the other hand, much more details can be found
# when using

cashocs.log.set_log_level(cashocs.log.DEBUG)

# which will provide detailed information regarding most operations cashocs performs.
# For a detailed introduction to the pythons logging
# library and the associated log levels, we refer the reader to
# <https://docs.python.org/3/library/logging.html>.
#
# As the output in the console might be hard to read, analyze or keep, cashocs offers
# the possibility to also log to various log files with the help of the
# {py:func}`cashocs.log.add_logfile` function, which is invoked as follows:

cashocs.log.add_logfile("output.log", mode="a", level=cashocs.log.DEBUG)

# This creates a new file named `output.log` which will be appended to if it is used
# multiple times, and which has a possibly different log level than the default terminal
# logger. This function can be called multiple times to define various log files.
# Alternatively, the mode {python}`mode="w"` can be used to overwrite the log file if it
# exists.
#
# Additionally, cashocs offers the possibility to add timestamps to the logs. These can
# be added with the command

cashocs.log.add_timestamps()

# so that you can see when an event was logged. If you have turned this on (it is
# disabled by default), the timestamps can be removed with a call to

cashocs.log.remove_timestamps()

# Finally, you can attach additional handlers for the log with the function
# {py:func}`cashocs.log.add_handler`. For this, you can just pass any valid
# handler (see <https://docs.python.org/3/library/logging.html#handler-objects>)
# and cashocs will also log to this handler.
