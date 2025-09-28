# ==============================================================================
# Refactored Python Code Examples
#
# This file contains a collection of refactored Python code snippets,
# demonstrating best practices such as:
# - PEP 8 styling and naming conventions.
# - Function encapsulation with clear inputs and outputs.
# - Descriptive docstrings and type hints.
# - Use of Pythonic idioms and efficient algorithms.
# - Example usage within `if __name__ == "__main__"` blocks.
# ==============================================================================

import math
import os
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from collections.abc import Iterable
from datetime import date, datetime
from itertools import groupby, permutations, product
from statistics import mode
from typing import (Any, Dict, Generator, List, Optional, Set, Tuple, Union)

# ==============================================================================
# 20. List Subset Check
# ==============================================================================

def is_subset(sub_list: List[Any], main_list: List[Any]) -> bool:
    """Checks if one list is a subset of another using sets for efficiency.

    Args:
        sub_list: The list to check if it's a subset.
        main_list: The list to check against.

    Returns:
        True if all elements of sub_list are in main_list, False otherwise.
    """
    return set(sub_list).issubset(set(main_list))

# Original, less efficient approach for comparison:
# def is_subset_original(sub_list: List[Any], main_list: List[Any]) -> bool:
#     """Checks if a list is a subset by iterating through elements."""
#     return all(item in main_list for item in sub_list)


# ==============================================================================
# 21. CamelCase String Check
# ==============================================================================

def is_camel_case(s: str) -> bool:
    """Checks if a string is in CamelCase.

    A string is considered CamelCase if it's not all lowercase, not all
    uppercase, and contains no underscores.

    Args:
        s: The input string.

    Returns:
        True if the string is CamelCase, False otherwise.
    """
    return s != s.lower() and s != s.upper() and "_" not in s


# ==============================================================================
# 22. Bytes Instance Check
# ==============================================================================

def is_bytes_instance(obj: Any) -> bool:
    """Checks if an object is an instance of bytes.

    Args:
        obj: The object to check.

    Returns:
        True if the object is a bytes instance, False otherwise.
    """
    return isinstance(obj, bytes)


# ==============================================================================
# 23. Find the Nth Prime Number
# ==============================================================================

def find_nth_prime(n_term: int) -> int:
    """Finds the nth prime number using a simple trial division method.

    Note: This implementation is simple but inefficient for very large n.

    Args:
        n_term: The position of the prime number to find (e.g., 5 for the 5th prime).

    Returns:
        The nth prime number.
        
    Raises:
        ValueError: If n_term is less than 1.
    """
    if n_term < 1:
        raise ValueError("Input must be a positive integer.")
    
    prime_count = 0
    num_to_check = 1
    while prime_count < n_term:
        num_to_check += 1
        is_prime = True
        # Check for factors from 2 up to the square root of the number
        for i in range(2, int(math.sqrt(num_to_check)) + 1):
            if num_to_check % i == 0:
                is_prime = False
                break
        if is_prime:
            prime_count += 1
    return num_to_check


# ==============================================================================
# 24. Temperature Conversion
# ==============================================================================

def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Converts a temperature from Fahrenheit to Celsius.

    Args:
        fahrenheit: The temperature in degrees Fahrenheit.

    Returns:
        The equivalent temperature in degrees Celsius.
    """
    return (fahrenheit - 32.0) * 5.0 / 9.0


# ==============================================================================
# 25. Decimal to Binary Conversion
# ==============================================================================

def decimal_to_binary_string(decimal_num: int) -> str:
    """Converts a decimal integer to its binary string representation.

    Args:
        decimal_num: The integer to convert.

    Returns:
        The binary string (e.g., '0b110').
    """
    return bin(decimal_num)


# ==============================================================================
# 26. Binary to Decimal Conversion (Manual Implementation)
# ==============================================================================

def binary_to_decimal(binary_str: str) -> int:
    """Converts a binary string to its base-10 integer equivalent.

    Note: The built-in `int(binary_str, 2)` is the standard way to do this.
    This function serves as a manual implementation example.

    Args:
        binary_str: The string representation of a binary number.

    Returns:
        The decimal (base-10) integer.
    """
    # A more Pythonic way to write the original logic:
    decimal_value = 0
    for i, digit in enumerate(reversed(binary_str)):
        if digit == '1':
            decimal_value += 2**i
    return decimal_value

    # The simplest, most idiomatic Python way:
    # return int(binary_str, 2)


# ==============================================================================
# 27. Execute Shell Command
# ==============================================================================

def execute_shell_command(command: List[str]) -> None:
    """Executes a shell command.

    Args:
        command: A list of strings representing the command and its arguments.
                 Example: ["sudo", "apt", "update"]
    """
    try:
        subprocess.run(command, check=True)
        print(f"Successfully executed: {' '.join(command)}")
    except FileNotFoundError:
        print(f"Error: Command '{command[0]}' not found.")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {' '.join(command)}\nReturn code: {e.returncode}")


# ==============================================================================
# 27. Generate a Random Integer in a Range
# ==============================================================================

def generate_random_int(min_val: int, max_val: int) -> int:
    """Generates a random integer within a specified inclusive range.

    Args:
        min_val: The minimum possible value.
        max_val: The maximum possible value.

    Returns:
        A random integer between min_val and max_val (inclusive).
    """
    return random.randint(min_val, max_val)


# ==============================================================================
# 28. Get a Random Element from a List
# ==============================================================================

def get_random_element(items: List[Any]) -> Any:
    """Selects a random element from a list.

    Args:
        items: The list from which to choose an element.

    Returns:
        A randomly selected element from the list.
    """
    if not items:
        return None
    return random.choice(items)


# ==============================================================================
# 29. Get Current Date
# ==============================================================================

def get_current_date() -> date:
    """Gets the current local date.

    Returns:
        A datetime.date object representing the current date.
    """
    return date.today()


# ==============================================================================
# 30. Count CPU Cores
# ==============================================================================

def get_cpu_count() -> int:
    """Gets the number of available CPU cores.

    Returns:
        The number of CPUs, or None if it cannot be determined.
    """
    return os.cpu_count()


# ==============================================================================
# 30. Remove Falsy Values from a List
# ==============================================================================

def remove_falsy_values(items: List[Any]) -> List[Any]:
    """Removes all "falsy" values from a list (e.g., None, 0, '', [], {}).

    Args:
        items: The list to filter.

    Returns:
        A new list with all falsy values removed.
    """
    return [item for item in items if item]


# ==============================================================================
# 31. Find the Most Common Element in a List
# ==============================================================================

def find_most_common_element(items: List[Any]) -> Any:
    """Finds the most frequently occurring element in a list.

    Args:
        items: A list of elements.

    Returns:
        The most common element.
    """
    return mode(items)


# ==============================================================================
# 32. Get Python Version
# ==============================================================================

def get_python_version() -> str:
    """Gets the current Python version string.

    Returns:
        A string containing the Python version information.
    """
    return sys.version


# ==============================================================================
# 33. Flatten a Nested List
# ==============================================================================

def flatten_list(nested_list: List[Any]) -> Generator[Any, None, None]:
    """Flattens a nested list of arbitrary depth.

    Args:
        nested_list: A list that may contain other lists as elements.

    Yields:
        The elements of the nested list in a flattened sequence.
    """
    for item in nested_list:
        if isinstance(item, Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten_list(item)
        else:
            yield item


# ==============================================================================
# 34. Parse XML from a String
# ==============================================================================

def parse_xml_from_string(xml_string: str) -> Optional[ET.Element]:
    """Parses an XML string and returns the root element.

    Args:
        xml_string: A string containing XML data.

    Returns:
        The root ElementTree element, or None if parsing fails.
    """
    try:
        return ET.fromstring(xml_string)
    except ET.ParseError as e:
        print(f"Error parsing XML string: {e}")
        return None


# ==============================================================================
# 35. Parse XML from a File
# ==============================================================================

def parse_xml_from_file(filepath: str) -> Optional[ET.Element]:
    """Parses an XML file and returns the root element.

    Args:
        filepath: The path to the XML file.

    Returns:
        The root ElementTree element, or None if parsing fails or file not found.
    """
    try:
        tree = ET.parse(filepath)
        return tree.getroot()
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except ET.ParseError as e:
        print(f"Error parsing XML file '{filepath}': {e}")
        return None


# ==============================================================================
# 36. Parse Datetime from String
# ==============================================================================

def parse_datetime(datetime_str: str, fmt: str) -> Optional[datetime]:
    """Parses a string into a datetime object given a specific format.

    Args:
        datetime_str: The string to parse.
        fmt: The format code string (e.g., '%b %d %Y %I:%M%p').

    Returns:
        A datetime object, or None if parsing fails.
    """
    try:
        return datetime.strptime(datetime_str, fmt)
    except ValueError:
        print(f"Error: String '{datetime_str}' does not match format '{fmt}'")
        return None


# ==============================================================================
# 37. Print List Elements without Brackets/Commas
# ==============================================================================

def print_list_concatenated(items: List[Any]) -> None:
    """Prints all elements of a list concatenated together without separators.

    Args:
        items: The list to print.
    """
    print(*items, sep='')


# ==============================================================================
# 38. Read a Specific Line from a File
# ==============================================================================

def read_specific_line(filepath: str, line_number: int) -> Optional[str]:
    """Reads a specific line from a text file.

    Note: This is inefficient for large files as it reads the whole file.
    For large files, it's better to iterate line by line.

    Args:
        filepath: The path to the text file.
        line_number: The line to read (1-indexed).

    Returns:
        The content of the specified line, or None if the line does not exist
        or the file is not found.
    """
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if 0 < line_number <= len(lines):
                return lines[line_number - 1].strip()
            return None
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None


# ==============================================================================
# 39. Remove Digits from a String
# ==============================================================================

def remove_digits_from_string(s: str) -> str:
    """Removes all digit characters from a string.

    Args:
        s: The input string.

    Returns:
        A new string with all digits removed.
    """
    return ''.join([char for char in s if not char.isdigit()])


# ==============================================================================
# 40. Nth Fibonacci Number (Recursive)
# ==============================================================================

def recursive_fibonacci(n: int) -> int:
    """Calculates the nth Fibonacci number using recursion.

    Note: This is a classic but highly inefficient implementation (O(2^n))
    due to repeated calculations. It's suitable for demonstration purposes
    only. For practical use, an iterative or memoized approach is far better.

    Args:
        n: The position in the Fibonacci sequence (1-indexed).

    Returns:
        The nth Fibonacci number.
    """
    if n <= 0:
        raise ValueError("Input must be a positive integer.")
    if n == 1:
        return 0
    if n == 2:
        return 1
    return recursive_fibonacci(n - 1) + recursive_fibonacci(n - 2)


# ==============================================================================
# 42. Subtract Two Matrices
# ==============================================================================

Matrix = List[List[Union[int, float]]]

def subtract_matrices(matrix1: Matrix, matrix2: Matrix) -> Matrix:
    """Subtracts matrix2 from matrix1 element-wise.

    Args:
        matrix1: The first matrix (minuend).
        matrix2: The second matrix (subtrahend).

    Returns:
        A new matrix that is the result of the subtraction.

    Raises:
        ValueError: If the matrices do not have the same dimensions.
    """
    rows1, cols1 = len(matrix1), len(matrix1[0])
    rows2, cols2 = len(matrix2), len(matrix2[0])

    if rows1 != rows2 or cols1 != cols2:
        raise ValueError("Matrices must have the same dimensions for subtraction.")

    return [[matrix1[i][j] - matrix2[i][j] for j in range(cols1)] for i in range(rows1)]


# ==============================================================================
# 49. Merge Sort Algorithm
# ==============================================================================

def merge_sort(data: List[Union[int, float]]) -> None:
    """Sorts a list in place using the merge sort algorithm.

    Args:
        data: The list of numbers to be sorted.
    """
    if len(data) > 1:
        mid = len(data) // 2
        left_half = data[:mid]
        right_half = data[mid:]

        merge_sort(left_half)
        merge_sort(right_half)

        i = j = k = 0
        # Merge the two sorted halves back into the original list
        while i < len(left_half) and j < len(right_half):
            if left_half[i] < right_half[j]:
                data[k] = left_half[i]
                i += 1
            else:
                data[k] = right_half[j]
                j += 1
            k += 1

        # Copy any remaining elements
        while i < len(left_half):
            data[k] = left_half[i]
            i += 1
            k += 1

        while j < len(right_half):
            data[k] = right_half[j]
            j += 1
            k += 1

# ==============================================================================
# 99. Roman to Integer Conversion (Class-based)
# ==============================================================================

class RomanConverter:
    """A class to handle conversions between Roman numerals and integers."""

    def __init__(self):
        """Initializes the converter with mapping values."""
        self.rom_to_int_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        self.int_to_rom_map = [
            (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
            (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"),
            (1, "I")
        ]

    def to_int(self, s: str) -> int:
        """Converts a Roman numeral string to an integer.

        Args:
            s: The Roman numeral string (e.g., "IX", "MCMXCIV").

        Returns:
            The integer equivalent.
        """
        int_val = 0
        for i in range(len(s)):
            # Subtractive case (e.g., IV, IX, XL)
            if i > 0 and self.rom_to_int_map[s[i]] > self.rom_to_int_map[s[i - 1]]:
                # Add the current value and subtract the previous value twice
                # (once to cancel it out, once to subtract it).
                int_val += self.rom_to_int_map[s[i]] - 2 * self.rom_to_int_map[s[i - 1]]
            else:
                int_val += self.rom_to_int_map[s[i]]
        return int_val

    def to_roman(self, num: int) -> str:
        """Converts an integer to a Roman numeral string.

        Args:
            num: The integer to convert (must be between 1 and 3999).

        Returns:
            The Roman numeral string representation.
        """
        if not 0 < num < 4000:
            raise ValueError("Input must be an integer between 1 and 3999")
            
        roman_num = ''
        for val, syb in self.int_to_rom_map:
            # Append symbol while number is greater than or equal to the value
            while num >= val:
                roman_num += syb
                num -= val
        return roman_num

# ==============================================================================
# Main execution block for demonstrating the functions
# ==============================================================================

if __name__ == "__main__":
    # --- 20. is_subset ---
    print("--- 20. is_subset ---")
    list1 = [1, 2, 3]
    list2 = [1, 2, 3, 4, 5]
    list3 = [1, 6]
    print(f"Is {list1} a subset of {list2}? {is_subset(list1, list2)}")  # True
    print(f"Is {list3} a subset of {list2}? {is_subset(list3, list2)}")  # False

    # --- 23. find_nth_prime ---
    print("\n--- 23. find_nth_prime ---")
    print(f"The 1st prime is: {find_nth_prime(1)}")    # 2
    print(f"The 10th prime is: {find_nth_prime(10)}")  # 29
    
    # --- 24. fahrenheit_to_celsius ---
    print("\n--- 24. fahrenheit_to_celsius ---")
    print(f"32°F is {fahrenheit_to_celsius(32):.1f}°C") # 0.0°C
    print(f"212°F is {fahrenheit_to_celsius(212):.1f}°C") # 100.0°C

    # --- 26. binary_to_decimal ---
    print("\n--- 26. binary_to_decimal ---")
    print(f"Binary '1101' is {binary_to_decimal('1101')} in decimal.") # 13
    
    # --- 30. remove_falsy_values ---
    print("\n--- 30. remove_falsy_values ---")
    mixed_list = [1, 0, 'hello', '', None, [], [1, 2], False, True]
    print(f"Original list: {mixed_list}")
    print(f"Cleaned list: {remove_falsy_values(mixed_list)}")

    # --- 33. flatten_list ---
    print("\n--- 33. flatten_list ---")
    nested = [1, [2, 3], [4, [5, 6]], 7]
    flattened = list(flatten_list(nested))
    print(f"Flattened {nested} is {flattened}")

    # --- 40. recursive_fibonacci ---
    print("\n--- 40. recursive_fibonacci ---")
    print(f"The 10th Fibonacci number is: {recursive_fibonacci(10)}") # 34

    # --- 42. subtract_matrices ---
    print("\n--- 42. subtract_matrices ---")
    m1 = [[10, 20], [30, 40]]
    m2 = [[1, 2], [3, 4]]
    result_matrix = subtract_matrices(m1, m2)
    print(f"Result of matrix subtraction: {result_matrix}")

    # --- 49. merge_sort ---
    print("\n--- 49. merge_sort ---")
    unsorted_list = [54, 26, 93, 17, 77, 31, 44, 55, 20]
    print(f"Unsorted: {unsorted_list}")
    merge_sort(unsorted_list)
    print(f"Sorted:   {unsorted_list}")
    
    # --- 99. RomanConverter ---
    print("\n--- 99. RomanConverter ---")
    converter = RomanConverter()
    print(f"Roman 'MCMXCIV' is {converter.to_int('MCMXCIV')}")  # 1994
    print(f"Integer 2023 is {converter.to_roman(2023)}")    # MMXXIII
