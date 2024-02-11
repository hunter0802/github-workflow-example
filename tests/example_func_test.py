import pytest
from app.functions.example_func import is_odd

def test_odd_number():
    assert is_odd(3) == 0

def test_even_number():
    assert is_odd(4) == 1