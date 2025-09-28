# -*- coding: utf-8 -*-
"""
A collection of simple Python code examples, refactored for clarity,
readability, and adherence to standard practices (PEP 8).

Each example is encapsulated in a function with a descriptive docstring and
type hints. The main execution block demonstrates the usage of each function.
This file is intended to be a learning resource and a demonstration of
good Python coding practices for beginners.

Author: [Your Name/GitHub Handle]
Date: [Current Date]
"""

# =============================================================================
# 1. Imports
# =============================================================================
import math
import random
import string
import sys
import subprocess
from datetime import date, datetime
from collections.abc import Iterable
from itertools import permutations
from statistics import mode
from typing import List, Any, Tuple, Dict, Set, Union, Optional


# =============================================================================
# 2. Function Definitions
# =============================================================================

# -----------------------------------------------------------------------------
# Section A: List & Iterable Operations
# -----------------------------------------------------------------------------

def remove_even_numbers(numbers: List[int]) -> List[int]:
    """
    Removes even numbers from a list using a list comprehension.

    Args:
        numbers: A list of integers.

    Returns:
        A new list containing only the odd numbers from the input list.
    """
    return [num for num in numbers if num % 2 != 0]


def elementwise_operation(
    list1: List[float], list2: List[float], op: str
) -> List[float]:
    """
    Performs an element-wise arithmetic operation on two lists of numbers.

    Args:
        list1: The first list of numbers.
        list2: The second list of numbers (must be same length as list1).
        op: The operation to perform ('add', 'sub', 'mul', 'div').

    Returns:
        A new list containing the results of the operation.

    Raises:
        ValueError: If lists have different lengths or op is invalid.
    """
    if len(list1) != len(list2):
        raise ValueError("Input lists must have the same length.")

    operations = {
        'add': lambda a, b: a + b,
        'sub': lambda a, b: a - b,
        'mul': lambda a, b: a * b,
        'div': lambda a, b: a / b if b != 0 else float('inf'),
    }

    if op not in operations:
        raise ValueError("Invalid operation. Choose from 'add', 'sub', 'mul', 'div'.")

    return [operations[op](a, b) for a, b in zip(list1, list2)]


def make_all_positive(numbers: List[float]) -> List[float]:
    """
    Converts all numbers in a list to their absolute (positive) values.

    Args:
        numbers: A list of numbers.

    Returns:
        A new list with all numbers converted to positive.
    """
    return [abs(num) for num in numbers]


def make_all_negative(numbers: List[float]) -> List[float]:
    """
    Converts all numbers in a list to their negative absolute values.

    Args:
        numbers: A list of numbers.

    Returns:
        A new list with all numbers converted to negative.
    """
    return [-abs(num) for num in numbers]


def make_negative_zero(numbers: List[float]) -> List[float]:
    """
    Replaces all negative values in a list with zero.

    Args:
        numbers: A list of numbers.

    Returns:
        A new list with negative values replaced by 0.
    """
    return [num if num >= 0 else 0 for num in numbers]


def normalize_list(numbers: List[float]) -> List[float]:
    """
    Normalizes a list of numbers to a scale of 0 to 1.

    Args:
        numbers: A list of numbers.

    Returns:
        A new list with values normalized between 0 and 1.
    """
    min_val, max_val = min(numbers), max(numbers)
    if min_val == max_val:
        return [0.0] * len(numbers)
    return [(val - min_val) / (max_val - min_val) for val in numbers]


def find_index_of_max(numbers: List[Union[int, float]]) -> Optional[int]:
    """
    Finds the index of the first occurrence of the maximum value in a list.

    Args:
        numbers: A list of numbers.

    Returns:
        The index of the maximum value, or None if the list is empty.
    """
    if not numbers:
        return None
    return numbers.index(max(numbers))


def flatten_nested_list(nested_list: List[Any]) -> List[Any]:
    """
    Flattens a nested list of any depth into a single list.

    Note: For simple cases (one level of nesting), a list comprehension or
    `itertools.chain.from_iterable` is often more efficient.

    Args:
        nested_list: A list that may contain other lists.

    Yields:
        Items from the flattened list.
    """
    for item in nested_list:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten_nested_list(item)
        else:
            yield item


def is_subset(sub_list: List[Any], main_list: List[Any]) -> bool:
    """
    Checks if one list is a subset of another list.

    Args:
        sub_list: The potential subset list.
        main_list: The main list to check against.

    Returns:
        True if all elements of sub_list are in main_list, False otherwise.
    """
    return all(item in main_list for item in sub_list)


# -----------------------------------------------------------------------------
# Section B: String Operations
# -----------------------------------------------------------------------------

def reverse_string(text: str) -> str:
    """
    Reverses a string using slicing.

    Args:
        text: The string to reverse.

    Returns:
        The reversed string.
    """
    return text[::-1]


def sort_comma_separated_string(csv_string: str) -> str:
    """
    Sorts a comma-separated string alphabetically.

    Args:
        csv_string: A string of values separated by commas.

    Returns:
        A new comma-separated string with items sorted.
    """
    items = csv_string.split(',')
    items.sort()
    return ','.join(items)


def get_unique_words(sentence: str) -> Set[str]:
    """
    Finds all unique words in a sentence.

    Args:
        sentence: The input sentence.

    Returns:
        A set containing the unique words.
    """
    return set(sentence.lower().split())


def filter_words_by_length(sentence: str, min_len: int = 0, max_len: int = sys.maxsize) -> List[str]:
    """
    Filters words in a sentence based on their length.

    Args:
        sentence: The input sentence.
        min_len: The minimum length of words to keep.
        max_len: The maximum length of words to keep.

    Returns:
        A list of words that meet the length criteria.
    """
    words = sentence.split(' ')
    return [word for word in words if min_len <= len(word) <= max_len]


def get_punctuations_from_string(text: str) -> Set[str]:
    """
    Extracts all punctuation characters used in a string.

    Args:
        text: The input string.

    Returns:
        A set of punctuation characters found in the string.
    """
    return {char for char in text if char in string.punctuation}


def replace_words_with_length(sentence: str) -> str:
    """
    Replaces each word in a sentence with its length.

    Args:
        sentence: The input sentence.

    Returns:
        A string where each word is replaced by its character count.
    """
    words = sentence.split(' ')
    lengths = [str(len(word)) for word in words]
    return ' '.join(lengths)

def is_camel_case(text: str) -> bool:
    """
    Checks if a string is in CamelCase.
    A string is considered CamelCase if it's not all lowercase, not all
    uppercase, and contains no underscores.

    Args:
        text: The string to check.

    Returns:
        True if the string is CamelCase, False otherwise.
    """
    return text != text.lower() and text != text.upper() and "_" not in text


# -----------------------------------------------------------------------------
# Section C: Number & Math Operations
# -----------------------------------------------------------------------------

def reverse_integer(num: int) -> int:
    """
    Reverses the digits of an integer.

    Args:
        num: The integer to reverse.

    Returns:
        The integer with its digits reversed. Handles negative numbers.
    """
    sign = -1 if num < 0 else 1
    reversed_num_str = str(abs(num))[::-1]
    return sign * int(reversed_num_str)


def sum_first_n_numbers(n: int) -> int:
    """
    Calculates the sum of the first n natural numbers using a loop.

    Note: A more efficient formula is `n * (n + 1) // 2`.
          Or using Python's `sum(range(1, n + 1))`.

    Args:
        n: The number of integers to sum (must be positive).

    Returns:
        The sum of numbers from 1 to n.
    """
    total = 0
    current_num = n
    while current_num > 0:
        total += current_num
        current_num -= 1
    return total


def factorial(n: int) -> int:
    """
    Calculates the factorial of a number using a loop.

    Note: `math.factorial(n)` is the standard and preferred way.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial of n.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    fact = 1
    current_num = n
    while current_num > 0:
        fact *= current_num
        current_num -= 1
    return fact


def get_factors(n: int) -> List[int]:
    """
    Finds all factors of a given integer.

    Args:
        n: A positive integer.

    Returns:
        A list of factors of n.
    """
    if n <= 0:
        return []
    return [i for i in range(1, n + 1) if n % i == 0]


def is_product_even(num1: int, num2: int) -> bool:
    """
    Checks if the product of two numbers is even.

    Args:
        num1: The first number.
        num2: The second number.

    Returns:
        True if the product is even, False otherwise.
    """
    return (num1 * num2) % 2 == 0


def is_sum_even(num1: int, num2: int) -> bool:
    """
    Checks if the sum of two numbers is even.

    Args:
        num1: The first number.
        num2: The second number.

    Returns:
        True if the sum is even, False otherwise.
    """
    return (num1 + num2) % 2 == 0


def calculate_nth_fibonacci(n: int) -> int:
    """
    Calculates the nth Fibonacci number using recursion.
    (Note: This is inefficient for large n due to repeated calculations).

    Args:
        n: The position in the Fibonacci sequence (1-based index).

    Returns:
        The nth Fibonacci number.
    """
    if n <= 0:
        raise ValueError("Input must be a positive integer.")
    if n == 1:
        return 0
    if n == 2:
        return 1
    return calculate_nth_fibonacci(n - 1) + calculate_nth_fibonacci(n - 2)


def is_perfect_square(n: int) -> bool:
    """
    Checks if a number is a perfect square.

    Args:
        n: A non-negative integer.

    Returns:
        True if n is a perfect square, False otherwise.
    """
    if n < 0:
        return False
    if n == 0:
        return True
    sqrt_n = int(math.sqrt(n))
    return sqrt_n * sqrt_n == n


# -----------------------------------------------------------------------------
# Section D: Random Data Generation
# -----------------------------------------------------------------------------

def generate_random_sample(start: int, end: int, count: int) -> List[int]:
    """
    Generates a sample of unique random integers from a given range.

    Args:
        start: The beginning of the range (inclusive).
        end: The end of the range (exclusive).
        count: The number of random integers to generate.

    Returns:
        A list of unique random integers.
    """
    return random.sample(range(start, end), count)


def generate_conditional_random_sample(start: int, end: int, count: int, condition: str) -> List[int]:
    """
    Generates a sample of random integers from a range that meet a condition.

    Args:
        start: The beginning of the range (inclusive).
        end: The end of the range (exclusive).
        count: The number of random integers to generate.
        condition: 'even', 'odd', or 'divisible_by_4'.

    Returns:
        A list of random integers meeting the condition.
    """
    if condition == 'even':
        population = range(start + (start % 2), end, 2)
    elif condition == 'odd':
        population = range(start + (start % 2 == 0), end, 2)
    elif condition == 'divisible_by_4':
        first_multiple = start + (4 - start % 4) % 4
        population = range(first_multiple, end, 4)
    else:
        raise ValueError("Condition not supported.")

    return random.sample(population, count)


def generate_random_vowels(count: int) -> List[str]:
    """
    Generates a list of random vowels with replacement.

    Args:
        count: The number of random vowels to generate.

    Returns:
        A list of random vowels.
    """
    vowels = 'aeiou'
    return random.choices(vowels, k=count)


# -----------------------------------------------------------------------------
# Section E: Dictionaries and Sets
# -----------------------------------------------------------------------------

def create_cubed_dict(n: int) -> Dict[int, int]:
    """
    Creates a dictionary with keys from 1 to n and values as their cubes.

    Args:
        n: The upper limit for the keys (inclusive).

    Returns:
        A dictionary where values are the cubes of the keys.
    """
    return {i: i ** 3 for i in range(1, n + 1)}


def list_to_indexed_dict(items: List[Any]) -> Dict[int, Any]:
    """
    Converts a list to a dictionary where keys are indices and values are items.

    Args:
        items: The list to convert.

    Returns:
        A dictionary created from the list.
    """
    return {index: value for index, value in enumerate(items)}


# -----------------------------------------------------------------------------
# Section F: System and Miscellaneous
# -----------------------------------------------------------------------------

def get_memory_usage(variable: Any) -> int:
    """
    Gets the memory size of a Python object in bytes.

    Args:
        variable: The variable to measure.

    Returns:
        The size of the variable in bytes.
    """
    return sys.getsizeof(variable)


def execute_bash_command(command: List[str]):
    """
    Executes a shell command.

    Args:
        command: A list of strings representing the command and its arguments.
                 Example: ['ls', '-l']
    """
    try:
        print(f"Executing command: {' '.join(command)}")
        subprocess.run(command, check=True)
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")


# =============================================================================
# 3. Main Execution Block
# =============================================================================

if __name__ == "__main__":
    print("--- Running Python Code Examples ---\n")

    # --- Section A: List & Iterable Operations ---
    print("--- Section A: List & Iterable Operations ---")
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Original list: {nums}")
    print(f"List with evens removed: {remove_even_numbers(nums)}")
    
    list1, list2 = [1, 2, 3, 4], [5, 6, 7, 8]
    print(f"Element-wise addition of {list1} and {list2}: {elementwise_operation(list1, list2, 'add')}")
    print(f"Element-wise subtraction: {elementwise_operation(list1, list2, 'sub')}")

    neg_nums = [-1, 2, -3, 4, -5]
    print(f"Original list for sign change: {neg_nums}")
    print(f"All positive: {make_all_positive(neg_nums)}")
    print(f"All negative: {make_all_negative(neg_nums)}")
    print(f"Negatives made zero: {make_negative_zero(neg_nums)}")

    norm_list = [2, 4, 10, 6, 8, 4]
    print(f"Normalized list of {norm_list}: {normalize_list(norm_list)}")
    
    shuffled_list = [1, 2, 3, 4, 5]
    random.shuffle(shuffled_list)
    print(f"Shuffled list: {shuffled_list}")

    nested = [1, [2, 3], [4, [5, 6]], 7]
    flat = list(flatten_nested_list(nested))
    print(f"Flattening {nested}: {flat}")
    
    print("\n--- Section B: String Operations ---")
    print(f"Reversing 'hello': {reverse_string('hello')}")
    csv = "five,two,three,four,one"
    print(f"Sorting '{csv}': {sort_comma_separated_string(csv)}")
    sentence = "The quick brown fox jumps over the lazy dog dog"
    print(f"Unique words in sentence: {get_unique_words(sentence)}")
    print(f"Words longer than 3 chars: {filter_words_by_length(sentence, min_len=4)}")
    print(f"Words with 3 chars or less: {filter_words_by_length(sentence, max_len=3)}")
    punctuated = "Hello, world! How are you?"
    print(f"Punctuation in '{punctuated}': {get_punctuations_from_string(punctuated)}")

    print("\n--- Section C: Number & Math Operations ---")
    print(f"Reversing integer 12345: {reverse_integer(12345)}")
    print(f"Reversing integer -54321: {reverse_integer(-54321)}")
    print(f"Sum of first 10 numbers: {sum_first_n_numbers(10)}")
    print(f"Factorial of 5: {factorial(5)} (Standard way: {math.factorial(5)})")
    print(f"Factors of 100: {get_factors(100)}")
    print(f"Is product of 4 and 5 even? {is_product_even(4, 5)}")
    print(f"Is sum of 4 and 5 even? {is_sum_even(4, 5)}")
    print(f"Is 81 a perfect square? {is_perfect_square(81)}")
    print(f"Is 80 a perfect square? {is_perfect_square(80)}")
    
    print("\n--- Section D: Random Data Generation ---")
    print(f"5 random numbers (100-200): {generate_random_sample(100, 200, 5)}")
    print(f"5 random even numbers (10-100): {generate_conditional_random_sample(10, 100, 5, 'even')}")
    print(f"5 random odd numbers (100-200): {generate_conditional_random_sample(100, 200, 5, 'odd')}")
    print(f"5 random vowels: {generate_random_vowels(5)}")
    
    print("\n--- Section E: Dictionaries and Sets ---")
    print(f"Dictionary of cubes up to 10: {create_cubed_dict(10)}")
    
    my_set = {1, 2, 3}
    my_list_to_add = [3, 4, 5]
    my_set.update(my_list_to_add)
    print(f"Set updated with a list: {my_set}")
    
    list_for_dict = ['a', 'b', 'c']
    print(f"List converted to indexed dict: {list_to_indexed_dict(list_for_dict)}")

    print("\n--- Section F: System and Miscellaneous ---")
    my_string = "this is a test string"
    print(f"Memory usage of '{my_string}': {get_memory_usage(my_string)} bytes")
    print(f"Current Python version: {sys.version}")

    # Example of running a safe bash command (use with caution)
    # On Linux/macOS: execute_bash_command(['echo', 'Hello from subprocess!'])
    # On Windows: execute_bash_command(['cmd', '/c', 'echo', 'Hello from subprocess!'])

    print("\n--- Geometric Calculations ---")
    radius = 10
    print(f"Circumference of a circle with radius {radius}: {2 * math.pi * radius:.2f}")
    length, width = 10, 5
    print(f"Area of a rectangle ({length}x{width}): {length * width}")
