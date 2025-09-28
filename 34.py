# -*- coding: utf-8 -*-
"""
A collection of refactored Python code examples following best practices.

This script contains a series of simple Python programs, originally written as
standalone snippets, that have been refactored into well-documented, reusable
functions. The goal is to demonstrate standard Python conventions, improve
readability, and create a clean, maintainable codebase suitable for learning
and sharing on platforms like GitHub.

Key improvements include:
- Wrapping logic in functions with clear, descriptive names.
- Adding type hints for function signatures.
- Writing comprehensive docstrings (Google-style).
- Using Pythonic idioms (e.g., list comprehensions, built-in functions).
- Replacing manual loops with more efficient or readable alternatives.
- Centralizing execution in a `main()` function with an `if __name__ == "__main__":` block.
"""

import collections
import itertools
import math
import random
import re
import string
from datetime import date
from functools import reduce

# ==============================================================================
# 1. ALGORITHMS
# ==============================================================================

def euclidean_gcd(num1: int, num2: int) -> int:
    """
    Finds the greatest common divisor (GCD) of two integers using the
    Euclidean algorithm.

    Note: The standard library's `math.gcd()` is the preferred way to do this
    in production code. This implementation is for educational purposes.

    Args:
        num1: The first integer.
        num2: The second integer.

    Returns:
        The greatest common divisor of num1 and num2.
    """
    a, b = abs(num1), abs(num2)
    while a != 0 and b != 0:
        if a > b:
            a %= b
        else:
            b %= a
    # At the end of the loop, one variable is the GCD and the other is 0.
    return a + b

def bubble_sort(items: list) -> list:
    """
    Sorts a list of items in ascending order using the bubble sort algorithm.

    Note: This is an inefficient sorting algorithm (O(n^2)) and should not be
    used for large datasets. Python's built-in `list.sort()` or `sorted()`
    are highly optimized and should be used instead.

    Args:
        items: A list of sortable items.

    Returns:
        A new list containing the sorted items.
    """
    # Create a copy to avoid modifying the original list
    sorted_items = items[:]
    n = len(sorted_items)
    for i in range(n - 1):
        for j in range(n - i - 1):
            if sorted_items[j] > sorted_items[j + 1]:
                # Pythonic tuple-swapping is cleaner than using a temp variable
                sorted_items[j], sorted_items[j + 1] = sorted_items[j + 1], sorted_items[j]
    return sorted_items

# ==============================================================================
# 2. STRING MANIPULATION
# ==============================================================================

def extract_integers_from_string(text: str) -> list[int]:
    """
    Selects all integers from a string and returns them as a list.

    Refactoring Note: Using the `re` (regular expressions) module is far
    more robust and concise than manual character-by-character iteration.

    Args:
        text: The string to search for integers.

    Returns:
        A list of integers found in the string.
    """
    # `re.findall` finds all non-overlapping matches of the pattern.
    # `\d+` matches one or more digits.
    return [int(num) for num in re.findall(r'\d+', text)]

def expand_character_range(start_char: str, end_char: str) -> str:
    """
    Expands a character range, like "a" to "f", into a full string "abcdef".

    Args:
        start_char: The starting character of the range (e.g., 'b').
        end_char: The ending character of the range (e.g., 'e').

    Returns:
        A string containing all characters from start to end, inclusive.
    """
    if ord(start_char) > ord(end_char):
        return ""
    # Use a generator expression within ''.join() for efficiency
    return ''.join(chr(i) for i in range(ord(start_char), ord(end_char) + 1))

def get_case_percentages(text: str) -> dict[str, float]:
    """
    Determines the percentage of lowercase and uppercase letters in a string.

    Args:
        text: The input string.

    Returns:
        A dictionary with the percentage of 'lower' and 'upper' case letters.
        Returns empty percentages if the string has no letters.
    """
    if not text:
        return {'lower': 0.0, 'upper': 0.0}

    lower_count = sum(1 for char in text if char.islower())
    upper_count = sum(1 for char in text if char.isupper())
    total_letters = lower_count + upper_count

    if total_letters == 0:
        return {'lower': 0.0, 'upper': 0.0}

    return {
        'lower': (lower_count / total_letters) * 100,
        'upper': (upper_count / total_letters) * 100
    }

def find_longest_word(sentence: str) -> str:
    """
    Finds and returns the longest word in a sentence.
    If there's a tie, the first longest word encountered is returned.

    Args:
        sentence: A string containing one or more words.

    Returns:
        The longest word in the sentence.
    """
    words = sentence.split()
    if not words:
        return ""
    # The `max` function can take a `key` argument to determine
    # what to compare. Here, we compare the length of each word.
    return max(words, key=len)

def remove_whitespace(text: str) -> str:
    """
    Removes all whitespace characters from a string.

    Args:
        text: The input string.

    Returns:
        The string with all whitespace removed.
    """
    return "".join(text.split())

# ==============================================================================
# 3. LIST AND DATA STRUCTURE OPERATIONS
# ==============================================================================

def find_largest_two(numbers: list[int | float]) -> tuple[int | float, int | float]:
    """
    Returns the largest and second largest elements from a list.

    Args:
        numbers: A list of numbers. Must contain at least two elements.

    Returns:
        A tuple containing (largest, second_largest).

    Raises:
        ValueError: If the list contains fewer than two elements.
    """
    if len(numbers) < 2:
        raise ValueError("Input list must contain at least two elements.")

    # A more robust way is to sort the unique elements
    unique_sorted = sorted(list(set(numbers)), reverse=True)
    
    if len(unique_sorted) < 2:
        # Handles cases like [5, 5, 5]
        return unique_sorted[0], unique_sorted[0]
        
    return unique_sorted[0], unique_sorted[1]

def get_element_frequencies(items: list) -> collections.Counter:
    """
    Calculates the frequency of each element in a list.

    Refactoring Note: The original code already used the best tool for this,
    `collections.Counter`. This function just formalizes its use.

    Args:
        items: The list of items to count.

    Returns:
        A collections.Counter object (a dict subclass) mapping items to their frequencies.
    """
    return collections.Counter(items)

def get_list_permutations(items: list) -> list[tuple]:
    """
    Generates all possible permutations of a list.

    Refactoring Note: The original code correctly used `itertools.permutations`.
    This function wraps it for clarity and reusability.

    Args:
        items: The list of items to permute.

    Returns:
        A list of tuples, where each tuple is a unique permutation.
    """
    return list(itertools.permutations(items))

def remove_duplicates_preserve_order(items: list) -> list:
    """
    Removes duplicate items from a list while preserving the original order.

    Refactoring Note: `dict.fromkeys()` in Python 3.7+ is a concise and
    efficient way to achieve this.

    Args:
        items: The list from which to remove duplicates.

    Returns:
        A new list with unique elements in their original order.
    """
    return list(dict.fromkeys(items))

def find_second_smallest(numbers: list[int | float]) -> int | float:
    """
    Returns the second smallest number in a list.

    Args:
        numbers: A list of numbers. Must contain at least two unique elements.

    Returns:
        The second smallest number.

    Raises:
        ValueError: If the list does not have at least two unique elements.
    """
    unique_numbers = sorted(list(set(numbers)))
    if len(unique_numbers) < 2:
        raise ValueError("List must contain at least two unique elements.")
    return unique_numbers[1]

def separate_positives_and_negatives(numbers: list[int | float]) -> dict[str, list]:
    """
    Separates a list of numbers into positive and negative lists.

    Refactoring Note: Using list comprehensions is more Pythonic and readable
    than iterating and appending manually.

    Args:
        numbers: A list of numbers.

    Returns:
        A dictionary with 'positives' and 'negatives' lists.
    """
    return {
        'positives': [num for num in numbers if num > 0],
        'negatives': [num for num in numbers if num < 0]
    }
    
# ==============================================================================
# 4. NUMBER THEORY AND CHECKS
# ==============================================================================

def is_perfect_number(n: int) -> bool:
    """
    Checks if a number is a perfect number.

    A perfect number is a positive integer that is equal to the sum of its
    proper positive divisors (the sum of its positive divisors excluding
    the number itself).

    Args:
        n: The number to check.

    Returns:
        True if the number is perfect, False otherwise.
    """
    if n <= 1:
        return False
    
    divisor_sum = sum(i for i in range(1, n) if n % i == 0)
    return divisor_sum == n
    
def is_leap_year(year: int) -> bool:
    """
    Determines if a given year is a leap year.

    Args:
        year: The year to check.

    Returns:
        True if the year is a leap year, False otherwise.
    """
    # A year is a leap year if it is divisible by 4,
    # except for end-of-century years, which must be divisible by 400.
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def is_armstrong_number(num: int) -> bool:
    """
    Checks if a number is an Armstrong number.
    
    An Armstrong number is a number that is the sum of its own digits each
    raised to the power of the number of digits.
    This implementation assumes 3 digits (e.g., 153 = 1^3 + 5^3 + 3^3).
    A more general implementation is recommended for production use.
    
    Args:
        num: The number to check.
        
    Returns:
        True if the number is an Armstrong number (for 3 digits), False otherwise.
    """
    if num < 0:
        return False
    
    s = str(num)
    order = len(s)
    
    digit_sum = sum(int(digit) ** order for digit in s)
    
    return num == digit_sum
    
# ==============================================================================
# 5. MAIN DEMONSTRATION FUNCTION
# ==============================================================================

def main():
    """Main function to demonstrate the refactored code."""
    print("--- Refactored Python Examples ---")

    # 1. GCD Algorithm
    print("\n1. Greatest Common Divisor (GCD) of 54 and 24:")
    print(f"   Result: {euclidean_gcd(54, 24)}")
    print(f"   Using math.gcd(): {math.gcd(54, 24)}")

    # 2. Extract Integers from String
    print("\n2. Extracting integers from 'Hello 123 World 45 and 678':")
    print(f"   Result: {extract_integers_from_string('Hello 123 World 45 and 678')}")

    # 3. Expand Character Range
    print("\n3. Expanding character range from 'b' to 'h':")
    print(f"   Result: '{expand_character_range('b', 'h')}'")

    # 4. Find Largest Two Numbers
    my_numbers = [10, 50, 2, 90, 45, 90, 100]
    print(f"\n4. Finding largest two in {my_numbers}:")
    largest, second_largest = find_largest_two(my_numbers)
    print(f"   Largest: {largest}, Second Largest: {second_largest}")

    # 5. Element Frequencies
    my_list = [10, 10, 20, 10, 20, 40, 50, 50, 30]
    print(f"\n5. Frequencies of elements in {my_list}:")
    print(f"   Result: {get_element_frequencies(my_list)}")

    # 6. List Permutations
    print("\n6. Permutations of [1, 2, 3]:")
    print(f"   Result: {get_list_permutations([1, 2, 3])}")

    # 7. Remove Duplicates
    dup_list = [10, 20, 30, 20, 10, 50, 60, 40, 80, 50, 40]
    print(f"\n7. Removing duplicates from {dup_list} (preserving order):")
    print(f"   Result: {remove_duplicates_preserve_order(dup_list)}")

    # 8. Second Smallest
    print("\n8. Finding second smallest in [1, 2, -8, -2, 0]:")
    print(f"   Result: {find_second_smallest([1, 2, -8, -2, 0])}")

    # 9. Case Percentages
    test_string_case = "This Is a Test String With 75% Lowercase Letters."
    print(f"\n9. Case percentages for '{test_string_case}':")
    percentages = get_case_percentages(test_string_case)
    print(f"   Lower: {percentages['lower']:.2f}%, Upper: {percentages['upper']:.2f}%")

    # 10. Separate Positives and Negatives
    rand_list = [random.randint(-10, 10) for _ in range(10)]
    print(f"\n10. Separating positives and negatives from {rand_list}:")
    separated = separate_positives_and_negatives(rand_list)
    print(f"   Positives: {separated['positives']}")
    print(f"   Negatives: {separated['negatives']}")

    # 11. Swap Case
    swap_case_str = "PyThOn Is FuN"
    print(f"\n11. Swapping case for '{swap_case_str}':")
    print(f"   Result: '{swap_case_str.swapcase()}'") # `swapcase` is a built-in method

    # 12. Bubble Sort
    unsorted_list = [64, 34, 25, 12, 22, 11, 90]
    print(f"\n12. Bubble sorting {unsorted_list}:")
    print(f"   Result: {bubble_sort(unsorted_list)}")

    # 13. Perfect Number Check
    print("\n13. Checking for perfect numbers:")
    print(f"   Is 28 a perfect number? {is_perfect_number(28)}")
    print(f"   Is 29 a perfect number? {is_perfect_number(29)}")

    # 14. Find Longest Word
    sentence = "Python is a high-level general-purpose programming language"
    print(f"\n14. Finding longest word in '{sentence}':")
    print(f"   Result: '{find_longest_word(sentence)}'")

    # 15. Dictionary Keys and Values
    my_dict = {'a': 1, 'b': 2, 'c': 3, 'd': 4}
    print(f"\n15. Keys and values of {my_dict}:")
    print(f"   Keys: {list(my_dict.keys())}")
    print(f"   Values: {list(my_dict.values())}")

    # 16. Squaring list items
    sample_list = [1, 2, 3, 4, 5, 6, 7]
    print(f"\n16. Squaring items in {sample_list}:")
    # List comprehension is the idiomatic way to do this.
    squared_list = [x**2 for x in sample_list]
    print(f"   Result: {squared_list}")
    
    # 17. Set Union
    set1 = {10, 20, 30, 40, 50}
    set2 = {30, 40, 50, 60, 70}
    print(f"\n17. Union of {set1} and {set2}:")
    print(f"   Result: {set1.union(set2)}") # or `set1 | set2`

    # 18. Fibonacci Series (Recursive example)
    def fibonacci_recursive(n: int) -> int:
        """Calculates the nth Fibonacci number recursively."""
        if n <= 1:
            return n
        return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)
    
    print("\n18. 10th Fibonacci number (recursive):")
    print(f"   Result: {fibonacci_recursive(10)}")

    # 19. Check if a number is a prime number
    def is_prime(n: int) -> bool:
        """Checks if a number is prime."""
        if n <= 1:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True

    print("\n19. Prime number check:")
    print(f"   Is 29 prime? {is_prime(29)}")
    print(f"   Is 407 prime? {is_prime(407)}") # 407 = 11 * 37

if __name__ == "__main__":
    main()
