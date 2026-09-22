import pytest

from app.tools.calculator import CalculatorError, calculate


def test_basic_arithmetic():
    assert calculate("2 + 3 * 4") == 14


def test_parentheses_and_division():
    assert calculate("(10 - 4) / 2") == 3


def test_average_helper():
    assert calculate("average(0.31, 0.28, 0.35)") == pytest.approx(0.31333333, rel=1e-5)


def test_power_and_modulo():
    assert calculate("2 ** 5") == 32
    assert calculate("10 % 3") == 1


def test_rejects_unsafe_expression():
    with pytest.raises(CalculatorError):
        calculate("__import__('os').system('echo hi')")


def test_rejects_garbage_input():
    with pytest.raises(CalculatorError):
        calculate("not a number")
