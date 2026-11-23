"""Заготовки задач на базовый Python."""

import math

from grader_contracts.python_basics import PositiveIntegerInput, TextInput, VectorPairInput


def count_vowels(data: TextInput) -> int:
    text = data.value
    vowels = set("aeiou")
    return sum(1 for char in text.lower() if char in vowels)


def has_unique_characters(data: TextInput) -> bool:
    text = data.value
    return len(set(text)) == len(text)


def count_one_bits(data: PositiveIntegerInput) -> int:
    number = data.value
    return bin(number).count("1")


def multiplicative_persistence(data: PositiveIntegerInput) -> int:
    number = data.value
    steps = 0
    while number >= 10:
        number = math.prod(int(digit) for digit in str(number))
        steps += 1
    return steps


def mse(data: VectorPairInput) -> float:
    predicted, expected = data.predicted, data.expected
    total = sum((p - e) ** 2 for p, e in zip(predicted, expected))
    return total / len(predicted)


def prime_factorization(data: PositiveIntegerInput) -> str:
    number = data.value
    factors: dict[int, int] = {}
    divisor = 2
    while divisor * divisor <= number:
        while number % divisor == 0:
            factors[divisor] = factors.get(divisor, 0) + 1
            number //= divisor
        divisor += 1
    if number > 1:
        factors[number] = factors.get(number, 0) + 1

    parts = [
        f"({base})" if power == 1 else f"({base}**{power})"
        for base, power in sorted(factors.items())
    ]
    return "".join(parts)


def pyramid(data: PositiveIntegerInput) -> int | str:
    cube_count = data.value
    accumulated = 0
    level = 0
    while accumulated < cube_count:
        level += 1
        accumulated += level * level
    if accumulated == cube_count:
        return level
    return "It is impossible"


def is_balanced_number(data: PositiveIntegerInput) -> bool:
    number = data.value
    digits = [int(d) for d in str(number)]
    half = (len(digits) - 1) // 2
    left_sum = sum(digits[:half])
    right_sum = sum(digits[len(digits) - half:])
    return left_sum == right_sum
