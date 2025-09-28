# -*- coding: utf-8 -*-
"""
A collection of refactored Python code examples covering various topics
including list manipulation, math functions, random number generation,
and basic algorithms.
"""

# =============================================================================
# 1. IMPORTS
# All imports are consolidated at the top for clarity and PEP 8 compliance.
# =============================================================================
import collections
import itertools
import math
import random
import statistics
import time
from timeit import Timer
from typing import Any, Dict, List, Set, Tuple, Union, Optional

# =============================================================================
# 2. FUNCTION AND CLASS DEFINITIONS
# Each function and class has been improved with docstrings, type hints,
# and Pythonic logic.
# =============================================================================

# Problem 55: Pad a list to a given length.
def pad_list_at_end(
    data: List[Any],
    target_length: int,
    pad_value: Any = 0
) -> List[Any]:
    """Pads a list with a value at the end to a specified length.

    Args:
        data: The input list.
        target_length: The desired length of the list.
        pad_value: The value to use for padding. Defaults to 0.

    Returns:
        A new list padded to the target length. Returns the original
        list if its length is already >= target_length.
    """
    padding_needed = target_length - len(data)
    if padding_needed <= 0:
        return data[:]  # Return a copy
    return data + [pad_value] * padding_needed

def pad_list_at_start(
    data: List[Any],
    target_length: int,
    pad_value: Any = 0
) -> List[Any]:
    """Pads a list with a value at the start to a specified length.

    Args:
        data: The input list.
        target_length: The desired length of the list.
        pad_value: The value to use for padding. Defaults to 0.

    Returns:
        A new list padded to the target length. Returns the original
        list if its length is already >= target_length.
    """
    padding_needed = target_length - len(data)
    if padding_needed <= 0:
        return data[:]  # Return a copy
    return [pad_value] * padding_needed + data

# Problem 56: Sigmoid activation function.
def sigmoid(x: float) -> float:
    """Implements the sigmoid activation function."""
    return 1 / (1 + math.exp(-x))

# Problem 57: Tanh activation function.
def tanh(x: float) -> float:
    """Implements the hyperbolic tangent (tanh) activation function."""
    return math.tanh(x)  # Using math.tanh is simpler and more direct.

# Problem 58: Area of an ellipse.
class Ellipse:
    """Represents an ellipse with major and minor axes."""
    def __init__(self, minor_axis: float, major_axis: float):
        """Initializes the Ellipse.

        Args:
            minor_axis: The length of the minor axis.
            major_axis: The length of the major axis.
        """
        self.minor_axis = minor_axis
        self.major_axis = major_axis

    def area(self) -> float:
        """Calculates the area of the ellipse."""
        return math.pi * self.major_axis * self.minor_axis

# Problem 59: Loop with a time delay.
def print_loop_with_delay(
    n: int = 10,
    delay_seconds: float = 1.0
) -> None:
    """Prints numbers from 0 to n-1 with a delay between each."""
    for i in range(n):
        print(i)
        time.sleep(delay_seconds)

# Problem 60: Get unique tokens (words) from a string.
def get_unique_words(text: str) -> Set[str]:
    """Returns the unique words (tokens) from a string.

    Args:
        text: The input string.

    Returns:
        A set of unique words.
    """
    return set(text.split())

# Problem 61: Standard deviation of a list.
def calculate_standard_deviation(numbers: List[float]) -> float:
    """Calculates the sample standard deviation of a list of numbers.

    Note: The standard library `statistics.stdev` is preferred for this.

    Args:
        numbers: A list of numbers.

    Returns:
        The sample standard deviation.
    """
    if len(numbers) < 2:
        raise ValueError("Standard deviation requires at least two data points.")
    mean = sum(numbers) / len(numbers)
    sum_sq_dev = sum((x - mean) ** 2 for x in numbers)
    return math.sqrt(sum_sq_dev / (len(numbers) - 1))

# Problem 62: Mode of a dataset.
def find_mode(data: List[Any]) -> Any:
    """Finds the mode (most common value) of a list.
    Uses the standard library `statistics.mode`.
    """
    return statistics.mode(data)

# Problem 63: Check if all numbers in a list are negative.
def are_all_negative(numbers: List[Union[int, float]]) -> bool:
    """Checks if all numbers in a list are negative.

    Returns:
        True if all numbers are negative, False otherwise.
    """
    if not numbers:
        return True # Or False, depending on convention for empty lists.
    return all(num < 0 for num in numbers)

# Problem 64: Check if numbers in a list sum to one.
def does_sum_equal_one(numbers: List[float]) -> bool:
    """Checks if all numbers in a list sum up to 1.0 using a tolerance.

    Args:
        numbers: A list of floats.

    Returns:
        True if the sum is close to 1.0, False otherwise.
    """
    return math.isclose(sum(numbers), 1.0)

# Problem 71: Length of a string.
def get_string_length(text: str) -> int:
    """Returns the length of a string."""
    return len(text)

# Problem 83: Remove duplicates from a list while preserving order.
def remove_duplicates_preserve_order(data: List[Any]) -> List[Any]:
    """Removes duplicate items from a list while preserving original order.

    A more modern and concise way for Python 3.7+ is: `list(dict.fromkeys(data))`
    """
    seen = set()
    result = []
    for item in data:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

# Problem 84: Person class hierarchy.
class Person:
    """A base class representing a person."""
    def get_gender(self) -> str:
        """Returns the gender of the person."""
        return "Unknown"

class Male(Person):
    """A class representing a male, inheriting from Person."""
    def get_gender(self) -> str:
        """Returns 'Male'."""
        return "Male"

class Female(Person):
    """A class representing a female, inheriting from Person."""
    def get_gender(self) -> str:
        """Returns 'Female'."""
        return "Female"

# Problem 86: Count words in a string.
def count_words(text: str) -> int:
    """Counts the number of words in a string."""
    return len(text.split())

# Problem 87: Get characters at even indices.
def get_even_index_chars(text: str) -> str:
    """Returns characters of a string that are at an even index."""
    return text[::2]

# Problem 89: Classic chicken and rabbit puzzle solver.
def solve_chicken_rabbit_puzzle(
    num_heads: int,
    num_legs: int
) -> Optional[Tuple[int, int]]:
    """Solves the classic chicken and rabbit puzzle.

    Args:
        num_heads: Total number of heads.
        num_legs: Total number of legs.

    Returns:
        A tuple (num_chickens, num_rabbits) or None if no solution exists.
    """
    for num_chickens in range(num_heads + 1):
        num_rabbits = num_heads - num_chickens
        if (2 * num_chickens) + (4 * num_rabbits) == num_legs:
            return num_chickens, num_rabbits
    return None

# Problem 90 & 91: Rounding numbers.
def round_up_to_integer(num: float) -> int:
    """Rounds a number up to the nearest integer (ceiling)."""
    return math.ceil(num)

def round_down_to_integer(num: float) -> int:
    """Rounds a number down to the nearest integer (floor)."""
    return math.floor(num)

# Problem 92: Standard rounding.
def round_to_nearest(num: float) -> int:
    """Rounds a number to the nearest integer."""
    return round(num)

# Problem 93: One's complement of a number.
def ones_complement(num: int) -> int:
    """Calculates the one's complement of a non-negative integer."""
    if num == 0:
        return 0
    num_bits = math.floor(math.log2(num)) + 1
    return ((1 << num_bits) - 1) ^ num

# Problem 94 & 95: Binary and decimal conversion.
def decimal_to_binary(num: int) -> str:
    """Converts a decimal integer to its binary string representation."""
    return bin(num)[2:]

def binary_to_decimal(binary_str: str) -> int:
    """Converts a binary string to its decimal integer equivalent."""
    return int(binary_str, 2)

# Problem 96: Duplicate a number into a list.
def create_repeated_list(value: Any, n: int) -> List[Any]:
    """Creates a list by repeating a value n times."""
    return [value] * n

# Problem 97: Find the nearest square number.
def find_nearest_square(n: int) -> int:
    """Finds the perfect square nearest to a given integer."""
    if n < 0:
        raise ValueError("Input must be a non-negative number.")
    sqrt_n = round(math.sqrt(n))
    return sqrt_n ** 2

# Problem 98: Midpoint between two numbers.
def midpoint(a: float, b: float) -> float:
    """Calculates the midpoint between two numbers."""
    return (a + b) / 2

# Problem 99: Reverse a string.
def reverse_string(text: str) -> str:
    """Reverses a string."""
    return text[::-1]

# Problem 100: Check if a string is a palindrome.
def is_palindrome(text: str) -> bool:
    """
    Checks if a string is a palindrome (reads the same forwards and backwards).
    The check is case-insensitive.
    """
    normalized_text = text.lower()
    return normalized_text == normalized_text[::-1]

# --- Unnumbered Functions from the Second Section ---

def add_two_numbers(num1: float, num2: float) -> float:
    """Adds two numbers and returns the sum."""
    return num1 + num2

def find_largest_of_three(
    num1: float,
    num2: float,
    num3: float
) -> float:
    """Finds and returns the largest among three numbers."""
    return max(num1, num2, num3)

def celsius_to_fahrenheit(celsius: float) -> float:
    """Converts a temperature from Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def fahrenheit_to_kelvin(fahrenheit: float) -> float:
    """Converts a temperature from Fahrenheit to Kelvin."""
    return (fahrenheit - 32) * 5/9 + 273.15

def celsius_to_kelvin(celsius: float) -> float:
    """Converts a temperature from Celsius to Kelvin."""
    return celsius + 273.15

def radians_to_degrees(radians: float) -> float:
    """Converts an angle from radians to degrees."""
    return math.degrees(radians)

def print_rotated_matrix_180(matrix: List[List[Any]]) -> None:
    """Prints a 2D matrix rotated by 180 degrees."""
    if not matrix or not all(matrix):
        print("Empty or invalid matrix.")
        return
    height = len(matrix)
    for i in range(height - 1, -1, -1):
        row = matrix[i]
        print(*row[::-1])

def bitwise_left_rotate(n: int, d: int, int_bits: int = 32) -> int:
    """Performs a bitwise left rotation."""
    d = d % int_bits
    return (n << d) | (n >> (int_bits - d))

def bitwise_right_rotate(n: int, d: int, int_bits: int = 32) -> int:
    """Performs a bitwise right rotation."""
    d = d % int_bits
    mask = (1 << int_bits) - 1
    return ((n >> d) | (n << (int_bits - d))) & mask

def get_symmetric_difference(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Returns the symmetric difference of two lists (unique elements in each)."""
    set1 = set(list1)
    set2 = set(list2)
    return list(set1.symmetric_difference(set2))

# =============================================================================
# 3. SCRIPT EXECUTION
# The main block demonstrates the usage of the refactored functions.
# =============================================================================
if __name__ == "__main__":
    print("--- Running Refactored Python Examples ---\n")

    # Problem 55: Padding a list
    print("### Problem 55: Padding a list ###")
    original_list = [1, 2, 3, 4, 5]
    padded_end = pad_list_at_end(original_list, 10)
    padded_start = pad_list_at_start(original_list, 10)
    print(f"Original: {original_list}")
    print(f"Padded at end: {padded_end}")
    print(f"Padded at start: {padded_start}\n")

    # Problem 56 & 57: Activation Functions
    print("### Problem 56 & 57: Activation Functions ###")
    print(f"Sigmoid of 0.5: {sigmoid(0.5):.4f}")
    print(f"Tanh of 0.5: {tanh(0.5):.4f}\n")

    # Problem 58: Ellipse Area
    print("### Problem 58: Ellipse Area ###")
    ellipse = Ellipse(minor_axis=2, major_axis=10)
    print(f"Area of ellipse (2, 10): {ellipse.area():.4f}\n")

    # Problem 60: Unique words
    print("### Problem 60: Unique words ###")
    sentence = "the quick brown fox jumps over the lazy dog"
    print(f"Unique words in '{sentence}': {get_unique_words(sentence)}\n")

    # Problem 61: Standard Deviation
    print("### Problem 61: Standard Deviation ###")
    data_points = [1, 2, 3, 4, 5, 6]
    print(f"Std dev of {data_points}: {calculate_standard_deviation(data_points):.4f}")
    print(f"Std dev using statistics module: {statistics.stdev(data_points):.4f}\n")

    # Problem 65: Random even number between 0 and 10
    print("### Problem 65: Random even number ###")
    evens = [i for i in range(11) if i % 2 == 0]
    print(f"Random even number (0-10): {random.choice(evens)}\n")

    # Problem 66: Random number divisible by 5 and 7
    print("### Problem 66: Random number divisible by 5 & 7 ###")
    divisible_by_35 = [i for i in range(201) if i % 35 == 0]
    print(f"Random number (0-200) div by 5&7: {random.choice(divisible_by_35)}\n")

    # Problem 67 & 68: Generate random lists
    print("### Problem 67 & 68: Generate random lists ###")
    random_nums = random.sample(range(100, 201), 5)
    print(f"5 random numbers (100-200): {random_nums}")
    even_pool = range(100, 201, 2)
    random_evens = random.sample(even_pool, 5)
    print(f"5 random even numbers (100-200): {random_evens}\n")

    # Problem 72: Execution time
    print("### Problem 72: Execution time ###")
    t = Timer("1+1")
    # Running for a larger number of times to get a meaningful measurement
    exec_time = t.timeit(number=1_000_000)
    print(f"Time to execute '1+1' 1,000,000 times: {exec_time:.6f} seconds\n")

    # Problem 73 & 74: Shuffle a list
    print("### Problem 73 & 74: Shuffle a list ###")
    my_list = [3, 6, 7, 8]
    print(f"Original list: {my_list}")
    random.shuffle(my_list)
    print(f"Shuffled list: {my_list}\n")

    # Problem 75: Generate sentences
    print("### Problem 75: Generate sentences ###")
    subjects = ["I", "You"]
    verbs = ["Play", "Love"]
    objects = ["Hockey", "Football"]
    for subject, verb, obj in itertools.product(subjects, verbs, objects):
        print(f"{subject} {verb} {obj}.")
    print()

    # Problem 76-81: List comprehensions for filtering
    print("### Problem 76-81: List comprehensions for filtering ###")
    list_a = [5, 6, 77, 45, 22, 12, 24]
    print(f"Remove evens from {list_a}: {[x for x in list_a if x % 2 != 0]}")

    list_b = [12, 24, 35, 70, 88, 120, 155]
    print(f"Remove nums div by 5&7 from {list_b}: {[x for x in list_b if x % 5 != 0 or x % 7 != 0]}")

    list_c = [12, 24, 35, 70, 88, 120, 155]
    print(f"Remove elements at even indices from {list_c}: {[x for i, x in enumerate(list_c) if i % 2 != 0]}")
    print(f"Remove elements at indices 0,4,5 from {list_c}: {[x for i, x in enumerate(list_c) if i not in (0, 4, 5)]}")
    print(f"Remove value 24 from {list_c}: {[x for x in list_c if x != 24]}\n")

    # Problem 82: List intersection
    print("### Problem 82: List intersection ###")
    list1 = [1, 3, 6, 78, 35, 55]
    list2 = [12, 24, 35, 24, 88, 120, 155]
    intersection = list(set(list1) & set(list2))
    print(f"Intersection of {list1} and {list2}: {intersection}\n")

    # Problem 83: Remove duplicates
    print("### Problem 83: Remove duplicates ###")
    dup_list = [12, 24, 35, 24, 88, 120, 155, 88, 120, 155]
    print(f"List with duplicates: {dup_list}")
    print(f"Deduplicated (custom func): {remove_duplicates_preserve_order(dup_list)}")
    print(f"Deduplicated (Pythonic): {list(dict.fromkeys(dup_list))}\n")

    # Problem 85: Character count
    print("### Problem 85: Character count ###")
    text_to_count = "JRR Tolkien"
    char_counts = collections.Counter(text_to_count)
    print(f"Character counts for '{text_to_count}':")
    for char, count in char_counts.items():
        print(f"'{char}': {count}")
    print()

    # Problem 88: Permutations
    print("### Problem 88: Permutations ###")
    perm_items = [1, 2, 3]
    print(f"Permutations of {perm_items}: {list(itertools.permutations(perm_items))}\n")

    # Problem 89: Chicken and Rabbit Puzzle
    print("### Problem 89: Chicken and Rabbit Puzzle ###")
    heads, legs = 35, 94
    solution = solve_chicken_rabbit_puzzle(heads, legs)
    if solution:
        chickens, rabbits = solution
        print(f"Solution for {heads} heads, {legs} legs: {chickens} chickens, {rabbits} rabbits\n")
    else:
        print("No solution found for the puzzle.\n")

    # Problem 100: Palindrome check
    print("### Problem 100: Palindrome check ###")
    test_str = "Nitin"
    print(f"Is '{test_str}' a palindrome? {is_palindrome(test_str)}")
    test_str_2 = "Racecar"
    print(f"Is '{test_str_2}' a palindrome? {is_palindrome(test_str_2)}\n")

    # Unnumbered examples
    print("### Unnumbered Examples ###")
    print(f"Symmetric difference: {get_symmetric_difference([10, 15, 20, 25], [25, 40, 35])}")
    matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    print("Matrix rotated 180 degrees:")
    print_rotated_matrix_180(matrix)
