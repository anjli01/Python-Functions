# -*- coding: utf-8 -*-
"""
A collection of refactored, well-documented, and Pythonic utility functions.

This module provides a variety of simple, reusable functions for common tasks
involving data manipulation, mathematics, and file operations, refactored from
an initial set of code snippets.

Key improvements include:
- PEP 8 compliance for code style and readability.
- Descriptive function and variable names.
- Type hints for all function signatures.
- Comprehensive docstrings explaining purpose, parameters, and returns.
- Conversion of standalone scripts into reusable functions.
- Use of context managers (`with`) for safe file handling.
- Algorithmic improvements for better performance (e.g., using sets, counters).
- A dedicated `if __name__ == "__main__":` block for clear examples.
"""

# =============================================================================
# 1. Imports (Grouped and at the top)
# =============================================================================
import os
import re
import math
import json
import random
from datetime import datetime, date
from collections import Counter, deque
from typing import List, Dict, Any, Union, Tuple, Iterable, TypeVar, Optional

# Define a generic type variable for functions that work on sequences
T = TypeVar('T')


# =============================================================================
# 2. List and Collection Manipulation Functions
# =============================================================================

def pluck(list_of_dicts: List[Dict[str, Any]], key: str) -> List[Any]:
    """
    Extracts a list of values corresponding to a specified key from a list of
    dictionaries.

    Args:
        list_of_dicts: A list of dictionaries.
        key: The key to pluck the value from.

    Returns:
        A list of values corresponding to the key. Returns None for dicts
        where the key is not found.
    """
    return [d.get(key) for d in list_of_dicts]


def reverse_sequence(sequence: Union[str, List[T]]) -> Union[str, List[T]]:
    """
    Reverses a list or a string.

    Args:
        sequence: The list or string to be reversed.

    Returns:
        A reversed copy of the input sequence.
    """
    return sequence[::-1]


def find_common_elements(list_a: List[T], list_b: List[T]) -> List[T]:
    """
    Returns a list of elements that exist in both provided lists.
    Note: This implementation is more efficient than nested loops for large lists.

    Args:
        list_a: The first list.
        list_b: The second list.

    Returns:
        A new list containing elements common to both lists.
    """
    set_b = set(list_b)
    return [item for item in list_a if item in set_b]


def sort_by_indexes(items: List[T], indexes: List[int], reverse: bool = False) -> List[T]:
    """
    Sorts a list based on a corresponding list of indexes.

    Args:
        items: The list to sort.
        indexes: The list of indexes to sort by.
        reverse: If True, sorts in descending order.

    Returns:
        A new list sorted according to the indexes.
    """
    if len(items) != len(indexes):
        raise ValueError("Input list and indexes list must have the same length.")
    return [val for _, val in sorted(zip(indexes, items), key=lambda x: x[0], reverse=reverse)]


def chunk_list(items: List[T], size: int) -> List[List[T]]:
    """
    Chunks a list into smaller lists of a specified size.
    Note: Refactored from a map/lambda to a more readable list comprehension.

    Args:
        items: The list to chunk.
        size: The desired size of each chunk.

    Returns:
        A list of lists, where each sublist has a maximum size as specified.
    """
    if size <= 0:
        raise ValueError("Chunk size must be a positive integer.")
    return [items[i:i + size] for i in range(0, len(items), size)]


def every_nth(items: List[T], nth: int) -> List[T]:
    """
    Returns every nth element in a list.

    Args:
        items: The list to process.
        nth: The step (e.g., 2 for every 2nd element).

    Returns:
        A new list containing every nth element.
    """
    if nth <= 0:
        raise ValueError("Nth element must be a positive integer.")
    return items[nth - 1::nth]


def get_first_element(items: List[T]) -> Optional[T]:
    """
    Returns the first element of a list, or None if the list is empty.

    Args:
        items: The list.

    Returns:
        The first element or None.
    """
    return items[0] if items else None


def have_same_contents(list_a: List[T], list_b: List[T]) -> bool:
    """
    Checks if two lists contain the same elements, regardless of order.
    Note: This is far more efficient than the original version using `count`.

    Args:
        list_a: The first list.
        list_b: The second list.

    Returns:
        True if lists have the same contents, False otherwise.
    """
    return Counter(list_a) == Counter(list_b)


def rotate_list(items: List[T], offset: int) -> List[T]:
    """
    Rotates a list by n positions to the left. For right rotation, use a
    negative offset.

    Args:
        items: The list to rotate.
        offset: The number of positions to rotate left.

    Returns:
        A new, rotated list.
    """
    # Use deque for more efficient rotation on large lists
    d = deque(items)
    d.rotate(-offset)
    return list(d)


def transpose_matrix(matrix: List[List[T]]) -> List[Tuple[T, ...]]:
    """
    Transposes a 2D list (matrix).

    Args:
        matrix: A list of lists representing the matrix.

    Returns:
        The transposed matrix as a list of tuples.
    """
    return list(zip(*matrix))


def filter_empty_rows(matrix: List[List[Any]]) -> List[List[Any]]:
    """
    Filters out empty lists (rows) from a list of lists (matrix).

    Args:
        matrix: The matrix to filter.

    Returns:
        A new matrix with empty rows removed.
    """
    return [row for row in matrix if row]


def add_matrices(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
    """
    Adds two matrices of the same dimensions.

    Args:
        matrix_a: The first matrix.
        matrix_b: The second matrix.

    Returns:
        The resulting matrix sum.

    Raises:
        ValueError: If matrices have different dimensions.
    """
    if len(matrix_a) != len(matrix_b) or len(matrix_a[0]) != len(matrix_b[0]):
        raise ValueError("Matrices must have the same dimensions for addition.")
    
    return [[matrix_a[i][j] + matrix_b[i][j] for j in range(len(matrix_a[0]))] for i in range(len(matrix_a))]


def multiply_matrices(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
    """
    Multiplies two matrices (A * B).

    Args:
        matrix_a: The first matrix (m x n).
        matrix_b: The second matrix (n x p).

    Returns:
        The resulting product matrix (m x p).
    
    Raises:
        ValueError: If matrices have incompatible dimensions for multiplication.
    """
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])

    if cols_a != rows_b:
        raise ValueError(f"Incompatible shapes for multiplication: {rows_a}x{cols_a} and {rows_b}x{cols_b}")
    
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a): # or rows_b
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result


def find_two_sum_indices(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Given a list of integers, return indices of the two numbers such that they
    add up to a specific target. This uses an efficient O(n) algorithm.

    Args:
        nums: A list of integers.
        target: The target sum.

    Returns:
        A tuple containing the indices of the two numbers, or None if not found.
    """
    num_map = {}  # To store number and its index
    for i, num in enumerate(nums):
        complement = target - num
        if complement in num_map:
            return (num_map[complement], i)
        num_map[num] = i
    return None


def summarize_ranges(nums: List[int]) -> List[str]:
    """
    Given a sorted integer list without duplicates, return the summary of its ranges.
    Example: [0,1,2,4,5,7] -> ["0->2", "4->5", "7"]

    Args:
        nums: A sorted list of unique integers.

    Returns:
        A list of strings representing the ranges.
    """
    if not nums:
        return []

    ranges = []
    start = nums[0]

    for i in range(1, len(nums)):
        if nums[i] != nums[i-1] + 1:
            end = nums[i-1]
            if start == end:
                ranges.append(str(start))
            else:
                ranges.append(f"{start}->{end}")
            start = nums[i]
    
    # Add the last range
    if start == nums[-1]:
        ranges.append(str(start))
    else:
        ranges.append(f"{start}->{nums[-1]}")
    
    return ranges


def find_missing_element(full_list: List[T], partial_list: List[T]) -> Optional[T]:
    """
    Finds the missing element between two lists where one is a shuffled version
    of the other with one element removed.

    Args:
        full_list: The original list of elements.
        partial_list: The shuffled list with one element missing.

    Returns:
        The missing element, or None if lists are identical.
    """
    # This sum-based method only works for numbers.
    # Using Counter is more generic and robust.
    full_count = Counter(full_list)
    partial_count = Counter(partial_list)
    
    missing = full_count - partial_count
    # `missing.keys()` returns a view object, so we get the first element
    return next(iter(missing.keys()), None)


def merge_overlapping_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """
    Given a collection of intervals, merge all overlapping intervals.
    The input list is assumed to be sorted by the start of the interval.

    Args:
        intervals: A list of intervals, e.g., [[1,3],[2,6],[8,10]].

    Returns:
        A list of merged intervals.
    """
    if not intervals:
        return []

    # If not pre-sorted, uncomment the next line:
    # intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        
        if current_start <= last_end: # Overlap detected
            # Merge by updating the end of the last interval in the merged list
            merged[-1][1] = max(last_end, current_end)
        else:
            merged.append([current_start, current_end])
            
    return merged


# =============================================================================
# 3. Dictionary Manipulation Functions
# =============================================================================

def sort_dict_by_key(d: Dict, reverse: bool = False) -> Dict:
    """
    Sorts a dictionary by its keys.

    Args:
        d: The dictionary to sort.
        reverse: If True, sorts in descending order.

    Returns:
        A new dictionary sorted by key.
    """
    return dict(sorted(d.items(), reverse=reverse))


def sort_dict_by_value(d: Dict, reverse: bool = False) -> Dict:
    """
    Sorts a dictionary by its values.

    Args:
        d: The dictionary to sort.
        reverse: If True, sorts in descending order.

    Returns:
        A new dictionary sorted by value.
    """
    return dict(sorted(d.items(), key=lambda item: item[1], reverse=reverse))


# =============================================================================
# 4. String Manipulation Functions
# =============================================================================

def to_snake_case(s: str) -> str:
    """
    Converts a string to snake_case.
    Handles CamelCase, PascalCase, and kebab-case.

    Args:
        s: The string to convert.

    Returns:
        The snake_cased string.
    """
    s = re.sub(r'([A-Z][a-z]+)', r' \1', s)
    s = re.sub(r'([A-Z]+)', r' \1', s)
    s = s.replace('-', ' ')
    return '_'.join(s.split()).lower()


def capitalize(s: str, lower_rest: bool = False) -> str:
    """
    Capitalizes the first letter of a string.

    Args:
        s: The input string.
        lower_rest: If True, converts the rest of the string to lowercase.

    Returns:
        The capitalized string.
    """
    if not s:
        return ""
    first_char = s[0].upper()
    rest = s[1:].lower() if lower_rest else s[1:]
    return f"{first_char}{rest}"


def decapitalize(s: str, upper_rest: bool = False) -> str:
    """
    De-capitalizes the first letter of a string.

    Args:
        s: The input string.
        upper_rest: If True, converts the rest of the string to uppercase.

    Returns:
        The de-capitalized string.
    """
    if not s:
        return ""
    first_char = s[0].lower()
    rest = s[1:].upper() if upper_rest else s[1:]
    return f"{first_char}{rest}"


def count_lowercase_chars(s: str) -> int:
    """
    Counts the number of lowercase characters in a string.

    Args:
        s: The input string.

    Returns:
        The count of lowercase characters.
    """
    return sum(1 for char in s if char.islower())


def find_longest_substring_without_repeats(s: str) -> int:
    """
    Finds the length of the longest substring without repeating characters.
    This implementation uses the efficient "sliding window" technique (O(n)).

    Args:
        s: The input string.

    Returns:
        The length of the longest non-repeating substring.
    """
    char_map = {}
    max_length = 0
    start = 0
    for end, char in enumerate(s):
        if char in char_map and char_map[char] >= start:
            start = char_map[char] + 1
        char_map[char] = end
        max_length = max(max_length, end - start + 1)
    return max_length


# =============================================================================
# 5. Numerical and Mathematical Functions
# =============================================================================

def clamp_number(num: float, lower_bound: float, upper_bound: float) -> float:
    """
    Clamps a number within a specified range [min_val, max_val].

    Args:
        num: The number to clamp.
        lower_bound: The lower boundary of the range.
        upper_bound: The upper boundary of the range.

    Returns:
        The clamped number.
    """
    min_val = min(lower_bound, upper_bound)
    max_val = max(lower_bound, upper_bound)
    return max(min(num, max_val), min_val)


def radians_to_degrees(rad: float) -> float:
    """
    Converts an angle from radians to degrees.
    Uses `math.pi` for better precision than a hardcoded value.
    
    Args:
        rad: The angle in radians.

    Returns:
        The angle in degrees.
    """
    return (rad * 180.0) / math.pi


def sigmoid(x: float) -> float:
    """
    Calculates the sigmoid value for any real number.

    Args:
        x: A real number.

    Returns:
        The sigmoid value, between 0 and 1.
    """
    return 1 / (1 + math.exp(-x))


def to_roman_numeral(num: int) -> str:
    """
    Converts an integer to its Roman numeral representation.

    Args:
        num: An integer (typically between 1 and 3999).

    Returns:
        The Roman numeral as a string.
    """
    if not 0 < num < 4000:
        raise ValueError("Input must be an integer between 1 and 3999")

    lookup = [
        (1000, 'M'), (900, 'CM'), (500, 'D'), (400, 'CD'), (100, 'C'),
        (90, 'XC'), (50, 'L'), (40, 'XL'), (10, 'X'), (9, 'IX'), (5, 'V'),
        (4, 'IV'), (1, 'I')
    ]
    res = ''
    for val, roman in lookup:
        d, num = divmod(num, val)
        res += roman * d
    return res


def to_binary_string(n: int) -> str:
    """
    Returns the binary representation of a given integer as a string.
    Removes the '0b' prefix from Python's built-in `bin()` function.

    Args:
        n: The integer to convert.

    Returns:
        The binary string representation.
    """
    return bin(n)[2:]


def weighted_average(nums: List[float], weights: List[float]) -> float:
    """
    Calculates the weighted average of two or more numbers.

    Args:
        nums: A list of numbers.
        weights: A list of corresponding weights.

    Returns:
        The weighted average.
    """
    if not nums or not weights:
        return 0.0
    
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Sum of weights cannot be zero.")
        
    return sum(x * y for x, y in zip(nums, weights)) / total_weight


def get_prime_factors(n: int) -> Iterable[int]:
    """
    Yields the prime factors of a given number.
    Note: Refactored to be a generator, which is more memory-efficient.

    Args:
        n: The number to factorize.

    Yields:
        The prime factors of the number.
    """
    # Handle factor of 2
    while n % 2 == 0:
        yield 2
        n //= 2
    # Handle odd factors
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            yield i
            n //= i
    # If n is still a prime number greater than 2
    if n > 2:
        yield n


def sum_of_powers(end: int, power: int = 2, start: int = 1) -> int:
    """
    Returns the sum of powers of numbers in a given range.

    Args:
        end: The end of the range (inclusive).
        power: The power to raise each number to. Defaults to 2.
        start: The start of the range (inclusive). Defaults to 1.

    Returns:
        The sum of the powers.
    """
    return sum(i ** power for i in range(start, end + 1))


def smallest_multiple(n: int) -> int:
    """
    Calculates the smallest positive number that is evenly divisible by all
    of the numbers from 1 to n (Least Common Multiple).
    Note: Refactored to a much more efficient algorithm using GCD.

    Args:
        n: The upper limit of the range.

    Returns:
        The smallest multiple.
    """
    if n <= 0:
        raise ValueError("Input must be a positive integer.")
    lcm = 1
    for i in range(1, n + 1):
        lcm = (lcm * i) // math.gcd(lcm, i)
    return lcm

def reverse_integer(n: int) -> int:
    """
    Reverses the digits of an integer, preserving the sign.

    Args:
        n: The integer to reverse.

    Returns:
        The reversed integer.
    """
    sign = -1 if n < 0 else 1
    reversed_str = str(abs(n))[::-1]
    return sign * int(reversed_str)


# =============================================================================
# 6. Date and Time Functions
# =============================================================================

def days_diff(start_date: date, end_date: date) -> int:
    """
    Calculates the number of full days between two dates.

    Args:
        start_date: The starting date object.
        end_date: The ending date object.

    Returns:
        The difference in days as an integer.
    """
    return (end_date - start_date).days


def to_iso_date_string(d: datetime) -> str:
    """
    Converts a datetime object to its ISO 8601 string representation.

    Args:
        d: The datetime object.

    Returns:
        An ISO 8601 formatted string.
    """
    return d.isoformat()


def drop_microseconds(dt_obj: datetime) -> datetime:
    """
    Removes the microsecond part from a datetime object.

    Args:
        dt_obj: The input datetime object.

    Returns:
        A new datetime object with microseconds set to 0.
    """
    return dt_obj.replace(microsecond=0)


def unix_timestamp_to_datetime(timestamp: Union[str, int]) -> datetime:
    """
    Converts a Unix timestamp (seconds since epoch) to a datetime object.

    Args:
        timestamp: The Unix timestamp as a string or integer.

    Returns:
        The corresponding datetime object.
    """
    return datetime.fromtimestamp(int(timestamp))


# =============================================================================
# 7. File System and Color Conversion Functions
# =============================================================================

def rgb_to_hex(r: int, g: int, b: int) -> str:
    """
    Converts RGB color components to a hexadecimal color code.
    Note: Using an f-string is a more modern approach.

    Args:
        r: Red component (0-255).
        g: Green component (0-255).
        b: Blue component (0-255).

    Returns:
        The hexadecimal color string (e.g., 'FFFFFF').
    """
    return f'{r:02X}{g:02X}{b:02X}'


def count_lines_in_file(filepath: str) -> int:
    """
    Counts the number of lines in a text file efficiently without loading
    the entire file into memory.

    Args:
        filepath: The path to the text file.

    Returns:
        The number of lines in the file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return 0


def count_files_in_directory(dir_path: str) -> int:
    """
    Counts the number of files (not directories) in a given directory.

    Args:
        dir_path: The path to the directory.

    Returns:
        The number of files.
    """
    try:
        return sum(1 for item in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, item)))
    except FileNotFoundError:
        print(f"Error: Directory not found at {dir_path}")
        return 0


# =============================================================================
# 8. Sorting Algorithms (for demonstration)
# =============================================================================

def bubble_sort(items: List[int]) -> List[int]:
    """
    A simple implementation of the Bubble Sort algorithm.
    Note: This is generally inefficient for large lists (O(n^2)).

    Args:
        items: A list of numbers to sort.

    Returns:
        A new, sorted list.
    """
    # Create a copy to avoid modifying the original list
    sorted_items = items[:]
    n = len(sorted_items)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if sorted_items[j] > sorted_items[j + 1]:
                sorted_items[j], sorted_items[j + 1] = sorted_items[j + 1], sorted_items[j]
    return sorted_items


def odd_even_sort(items: List[int]) -> List[int]:
    """
    Implementation of the Odd-Even Sort (or Brick Sort) algorithm.
    
    Args:
        items: A list of numbers to sort.

    Returns:
        A new, sorted list.
    """
    sorted_items = items[:]
    n = len(sorted_items)
    is_sorted = False
    while not is_sorted:
        is_sorted = True
        # Odd phase
        for i in range(1, n - 1, 2):
            if sorted_items[i] > sorted_items[i + 1]:
                sorted_items[i], sorted_items[i + 1] = sorted_items[i + 1], sorted_items[i]
                is_sorted = False
        # Even phase
        for i in range(0, n - 1, 2):
            if sorted_items[i] > sorted_items[i + 1]:
                sorted_items[i], sorted_items[i + 1] = sorted_items[i + 1], sorted_items[i]
                is_sorted = False
    return sorted_items


# =============================================================================
# 9. Example Usage
# =============================================================================

def main():
    """Main function to demonstrate the usage of utility functions."""
    print("--- Running Demonstrations for python_utilities.py ---\n")

    # --- List and Collection Demos ---
    print("--- List and Collection Manipulation ---")
    users = [{'name': 'Alice', 'age': 30}, {'name': 'Bob', 'age': 25}]
    print(f"Pluck 'name' from users: {pluck(users, 'name')}")
    print(f"Reverse 'hello': {reverse_sequence('hello')}")
    print(f"Common elements in [1,2,3] and [3,4,5]: {find_common_elements([1,2,3], [3,4,5])}")
    print(f"Sort [a,b,c] by [2,3,1]: {sort_by_indexes(['a','b','c'], [2,3,1])}")
    print(f"Chunk [1-9] into size 3: {chunk_list(list(range(1, 10)), 3)}")
    print(f"Every 3rd element in [1-10]: {every_nth(list(range(1, 11)), 3)}")
    print(f"First element of [1,2,3]: {get_first_element([1,2,3])}")
    print(f"Do [1,2,3] and [3,2,1] have same contents? {have_same_contents([1,2,3], [3,2,1])}")
    print(f"Rotate [1,2,3,4,5] by 2: {rotate_list([1,2,3,4,5], 2)}")
    matrix = [[1, 2, 3], [4, 5, 6]]
    print(f"Transpose of {matrix}: {transpose_matrix(matrix)}")
    matrix_with_empty = [[1], [], [2, 3], []]
    print(f"Filter empty rows from {matrix_with_empty}: {filter_empty_rows(matrix_with_empty)}")
    print(f"Two Sum for [2, 7, 11, 15] with target 9: {find_two_sum_indices([2, 7, 11, 15], 9)}")
    print(f"Summarize ranges for [0,1,2,4,5,7]: {summarize_ranges([0,1,2,4,5,7])}")
    print(f"Missing element in [1..7] vs [3,7,2,1,4,6]: {find_missing_element([1,2,3,4,5,6,7], [3,7,2,1,4,6])}")
    intervals = [[1,3],[2,6],[8,10],[15,18]]
    print(f"Merge intervals {intervals}: {merge_overlapping_intervals(intervals)}")

    # --- Dictionary Demos ---
    print("\n--- Dictionary Manipulation ---")
    my_dict = {'c': 3, 'a': 1, 'b': 2}
    print(f"Original dict: {my_dict}")
    print(f"Sorted by key: {sort_dict_by_key(my_dict)}")
    print(f"Sorted by value: {sort_dict_by_value(my_dict)}")

    # --- String Demos ---
    print("\n--- String Manipulation ---")
    print(f"Snake case for 'someCamelCaseString': {to_snake_case('someCamelCaseString')}")
    print(f"Capitalize 'hello world': {capitalize('hello world')}")
    print(f"Longest substring in 'abcabcbb': {find_longest_substring_without_repeats('abcabcbb')}")

    # --- Math Demos ---
    print("\n--- Numerical and Mathematical ---")
    print(f"Clamp 5.5 between 1 and 5: {clamp_number(5.5, 1, 5)}")
    print(f"Pi radians to degrees: {radians_to_degrees(math.pi)}")
    print(f"Sigmoid of 0.5: {sigmoid(0.5)}")
    print(f"1994 to Roman numeral: {to_roman_numeral(1994)}")
    print(f"10 to binary: {to_binary_string(10)}")
    print(f"Weighted average of [10, 20] with weights [1, 3]: {weighted_average([10, 20], [1, 3])}")
    print(f"Prime factors of 100: {list(get_prime_factors(100))}")
    print(f"Sum of squares up to 3: {sum_of_powers(3, power=2)}")
    print(f"Smallest multiple for 1 to 10: {smallest_multiple(10)}")
    print(f"Reverse of -123: {reverse_integer(-123)}")
    
    # --- Date/Time Demos ---
    print("\n--- Date and Time ---")
    d1 = date(2023, 1, 1)
    d2 = date(2023, 1, 31)
    print(f"Days between {d1} and {d2}: {days_diff(d1, d2)}")
    now = datetime.now()
    print(f"Current time in ISO format: {to_iso_date_string(now)}")
    print(f"Current time without microseconds: {drop_microseconds(now)}")
    ts = "1284105682"
    print(f"Unix timestamp {ts} to datetime: {unix_timestamp_to_datetime(ts)}")

    # --- Matrix Demos ---
    print("\n--- Matrix Operations ---")
    X = [[1, 2], [3, 4]]
    Y = [[5, 6], [7, 8]]
    print(f"Matrix addition of {X} and {Y}: {add_matrices(X, Y)}")
    Y_mult = [[5, 6, 7], [8, 9, 10]]
    print(f"Matrix multiplication of {X} and {Y_mult}: {multiply_matrices(X, Y_mult)}")

    # --- Other Demos ---
    print("\n--- Miscellaneous ---")
    print(f"RGB(255, 0, 128) to Hex: #{rgb_to_hex(255, 0, 128)}")
    
    # File I/O Demo
    dummy_filepath = "temp_demo_file.txt"
    with open(dummy_filepath, "w") as f:
        f.write("First line\nSecond line\nThird line\n")
    print(f"Lines in '{dummy_filepath}': {count_lines_in_file(dummy_filepath)}")
    os.remove(dummy_filepath)

    print(f"Files in current directory ('.'): {count_files_in_directory('.')}")

    # Sorting Demo
    unsorted_list = [64, 34, 25, 12, 22, 11, 90]
    print(f"\nUnsorted list: {unsorted_list}")
    print(f"Bubble Sorted: {bubble_sort(unsorted_list)}")
    print(f"Odd-Even Sorted: {odd_even_sort(unsorted_list)}")
    
    # JSON Demo
    person_dict = {'name': 'Alice', 'age': 30, 'city': 'New York'}
    person_json = json.dumps(person_dict, indent=2)
    print(f"\nDictionary converted to JSON:\n{person_json}")


if __name__ == "__main__":
    main()
