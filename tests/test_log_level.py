"""
Created on 27/07/2021, 09.59

@author: blauths
"""

import cashocs
from cashocs._loggers import debug, info, warning, error, critical


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
