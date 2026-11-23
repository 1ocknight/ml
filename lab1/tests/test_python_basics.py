"""Тесты первой части лабораторной №1."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grader_contracts.python_basics import (
    PositiveIntegerInput,
    TextInput,
    VectorPairInput,
)
from python_basics import (
    count_one_bits,
    count_vowels,
    has_unique_characters,
    is_balanced_number,
    mse,
    multiplicative_persistence,
    prime_factorization,
    pyramid,
)


def test_count_vowels_ignores_case():
    assert count_vowels(TextInput("Hello World")) == 3
    assert count_vowels(TextInput("AEIOU")) == 5
    assert count_vowels(TextInput("xyz")) == 0
    assert count_vowels(TextInput("")) == 0


def test_has_unique_characters():
    assert has_unique_characters(TextInput("abc")) is True
    assert has_unique_characters(TextInput("")) is True
    assert has_unique_characters(TextInput("aba")) is False


def test_count_one_bits():
    assert count_one_bits(PositiveIntegerInput(13)) == 3
    assert count_one_bits(PositiveIntegerInput(1)) == 1
    assert count_one_bits(PositiveIntegerInput(255)) == 8


def test_multiplicative_persistence():
    assert multiplicative_persistence(PositiveIntegerInput(39)) == 3
    assert multiplicative_persistence(PositiveIntegerInput(4)) == 0
    assert multiplicative_persistence(PositiveIntegerInput(999)) == 4


def test_mse():
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 3])) == 0.0
    assert mse(VectorPairInput([1, 2, 3], [1, 2, 4])) == 1 / 3


def test_prime_factorization():
    assert prime_factorization(PositiveIntegerInput(86240)) == "(2**5)(5)(7**2)(11)"
    assert prime_factorization(PositiveIntegerInput(13)) == "(13)"
    assert prime_factorization(PositiveIntegerInput(8)) == "(2**3)"


def test_pyramid():
    assert pyramid(PositiveIntegerInput(14)) == 3
    assert pyramid(PositiveIntegerInput(1)) == 1
    assert pyramid(PositiveIntegerInput(10)) == "It is impossible"


def test_is_balanced_number():
    assert is_balanced_number(PositiveIntegerInput(1234006)) is True
    assert is_balanced_number(PositiveIntegerInput(123456)) is False
    assert is_balanced_number(PositiveIntegerInput(7)) is True
