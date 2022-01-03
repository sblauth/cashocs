# Copyright (C) 2020-2022 Sebastian Blauth
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

import cashocs
from cashocs._loggers import critical, debug, error, info, warning


def test_set_log_level(caplog):
    cashocs.set_log_level(cashocs.LogLevel.DEBUG)
    debug("abc")
    info("def")
    warning("ghi")
    error("jkl")
    critical("mno")
    assert "abc" in caplog.text
    assert "def" in caplog.text
    assert "ghi" in caplog.text
    assert "jkl" in caplog.text
    assert "mno" in caplog.text
    caplog.clear()

    cashocs.set_log_level(cashocs.LogLevel.INFO)
    debug("abc")
    info("def")
    warning("ghi")
    error("jkl")
    critical("mno")
    assert not "abc" in caplog.text
    assert "def" in caplog.text
    assert "ghi" in caplog.text
    assert "jkl" in caplog.text
    assert "mno" in caplog.text
    caplog.clear()

    cashocs.set_log_level(cashocs.LogLevel.WARNING)
    debug("abc")
    info("def")
    warning("ghi")
    error("jkl")
    critical("mno")
    assert not "abc" in caplog.text
    assert not "def" in caplog.text
    assert "ghi" in caplog.text
    assert "jkl" in caplog.text
    assert "mno" in caplog.text
    caplog.clear()

    cashocs.set_log_level(cashocs.LogLevel.ERROR)
    debug("abc")
    info("def")
    warning("ghi")
    error("jkl")
    critical("mno")
    assert not "abc" in caplog.text
    assert not "def" in caplog.text
    assert not "ghi" in caplog.text
    assert "jkl" in caplog.text
    assert "mno" in caplog.text
    caplog.clear()

    cashocs.set_log_level(cashocs.LogLevel.CRITICAL)
    debug("abc")
    info("def")
    warning("ghi")
    error("jkl")
    critical("mno")
    assert not "abc" in caplog.text
    assert not "def" in caplog.text
    assert not "ghi" in caplog.text
    assert not "jkl" in caplog.text
    assert "mno" in caplog.text
    caplog.clear()
