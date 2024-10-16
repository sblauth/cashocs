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

import fenics

import cashocs
from cashocs.log import critical
from cashocs.log import debug
from cashocs.log import error
from cashocs.log import info
from cashocs.log import warning


def issue_messages():
    debug("abc")
    info("def")
    warning("ghi")
    error("jkl")
    critical("mno")


def test_set_log_level(caplog):
    cashocs.set_log_level(cashocs.log.DEBUG)
    issue_messages()
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert "abc" in caplog.text
        assert "def" in caplog.text
        assert "ghi" in caplog.text
        assert "jkl" in caplog.text
        assert "mno" in caplog.text
    fenics.MPI.barrier(fenics.MPI.comm_world)
    caplog.clear()

    cashocs.set_log_level(cashocs.log.INFO)

    issue_messages()
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert not "abc" in caplog.text
        assert "def" in caplog.text
        assert "ghi" in caplog.text
        assert "jkl" in caplog.text
        assert "mno" in caplog.text
    fenics.MPI.barrier(fenics.MPI.comm_world)
    caplog.clear()

    cashocs.set_log_level(cashocs.log.WARNING)
    issue_messages()
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert not "abc" in caplog.text
        assert not "def" in caplog.text
        assert "ghi" in caplog.text
        assert "jkl" in caplog.text
        assert "mno" in caplog.text
    fenics.MPI.barrier(fenics.MPI.comm_world)
    caplog.clear()

    cashocs.set_log_level(cashocs.log.ERROR)
    issue_messages()
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert not "abc" in caplog.text
        assert not "def" in caplog.text
        assert not "ghi" in caplog.text
        assert "jkl" in caplog.text
        assert "mno" in caplog.text
    fenics.MPI.barrier(fenics.MPI.comm_world)
    caplog.clear()

    cashocs.set_log_level(cashocs.log.CRITICAL)
    issue_messages()
    if fenics.MPI.rank(fenics.MPI.comm_world) == 0:
        assert not "abc" in caplog.text
        assert not "def" in caplog.text
        assert not "ghi" in caplog.text
        assert not "jkl" in caplog.text
        assert "mno" in caplog.text
    fenics.MPI.barrier(fenics.MPI.comm_world)
    caplog.clear()
