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

import fenics

import cashocs
from cashocs._loggers import critical
from cashocs._loggers import debug
from cashocs._loggers import error
from cashocs._loggers import info
from cashocs._loggers import warning


def issue_messages():
    debug("abc")
    info("def")
    warning("ghi")
    error("jkl")
    critical("mno")


def test_set_log_level(caplog):
    cashocs.set_log_level(cashocs.LogLevel.DEBUG)
    issue_messages()
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert "abc" in caplog.text
        assert "def" in caplog.text
        assert "ghi" in caplog.text
        assert "jkl" in caplog.text
        assert "mno" in caplog.text
    fenics.MPI.barrier(fenics.MPI.comm_world)
    caplog.clear()

    cashocs.set_log_level(cashocs.LogLevel.INFO)

    issue_messages()
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert not "abc" in caplog.text
        assert "def" in caplog.text
        assert "ghi" in caplog.text
        assert "jkl" in caplog.text
        assert "mno" in caplog.text
    fenics.MPI.barrier(fenics.MPI.comm_world)
    caplog.clear()

    cashocs.set_log_level(cashocs.LogLevel.WARNING)
    issue_messages()
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert not "abc" in caplog.text
        assert not "def" in caplog.text
        assert "ghi" in caplog.text
        assert "jkl" in caplog.text
        assert "mno" in caplog.text
    fenics.MPI.barrier(fenics.MPI.comm_world)
    caplog.clear()

    cashocs.set_log_level(cashocs.LogLevel.ERROR)
    issue_messages()
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert not "abc" in caplog.text
        assert not "def" in caplog.text
        assert not "ghi" in caplog.text
        assert "jkl" in caplog.text
        assert "mno" in caplog.text
    fenics.MPI.barrier(fenics.MPI.comm_world)
    caplog.clear()

    cashocs.set_log_level(cashocs.LogLevel.CRITICAL)
    issue_messages()
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert not "abc" in caplog.text
        assert not "def" in caplog.text
        assert not "ghi" in caplog.text
        assert not "jkl" in caplog.text
        assert "mno" in caplog.text
    fenics.MPI.barrier(fenics.MPI.comm_world)
    caplog.clear()
