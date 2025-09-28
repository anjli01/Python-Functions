# -*- coding: utf-8 -*-
"""
A collection of refactored Python code examples for various tasks.

This script demonstrates best practices for writing clean, readable, and reusable
Python code, including:
- Adherence to PEP 8 style guide.
- Use of descriptive function and variable names.
- Comprehensive docstrings and type hints.
- Separation of logic (functions) from execution (main block).
- Utilization of Python's standard library and idiomatic patterns.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import math
import random
import datetime
from collections import defaultdict, Counter
import ast  # Abstract Syntax Trees for safe string evaluation


# =============================================================================
# 2. CONSTANTS
# =============================================================================
KILOMETERS_TO_MILES = 0.621371


# =============================================================================
# 3. GEOMETRY CALCULATIONS
# =============================================================================

def calculate_cylinder_csa(radius: float, height: float) -> float:
    """Calculates the Curved Surface Area (CSA) of a cylinder.

    Args:
        radius (float): The radius of the cylinder's base.
        height (float): The height of the cylinder.

    Returns:
        float: The curved surface area of the cylinder.
    """
    return 2 * math.pi * radius * height


def calculate_icosahedron_area(side_length: float) -> float:
    """Calculates the surface area of a regular icosahedron.

    Args:
        side_length (float): The length of one edge of the icosahedron.

    Returns:
        float: The total surface area.
    """
    return 5 * math.sqrt(3) * side_length ** 2


def calculate_icosahedron_volume(side_length: float) -> float:
    """Calculates the volume of a regular icosahedron.

    Args:
        side_length (float): The length of one edge of the icosahedron.

    Returns:
        float: The volume of the icosahedron.
    """
    phi = (1 + math.sqrt(5)) / 2  # Golden ratio
    return (5 / 12) * (3 + math.sqrt(5)) * side_length ** 3
    # An alternative calculation using the golden ratio:
    # return (5 * phi**2 / 6) * side_length**3


def calculate_pentagonal_prism_surface_area(
    side_length: float, base_apothem: float, height: float
) -> float:
    """Calculates the surface area of a pentagonal prism.
    
    Note: The original formula `5*a*b + 5*b*h` seems specific to a certain
    interpretation of 'a', 'b', and 'h'. A more standard formula is used here.
    Area = 2 * (Base Area) + 5 * (Side Face Area)
    Base Area = 5/2 * side_length * base_apothem

    Args:
        side_length (float): The length of a side of the pentagonal base.
        base_apothem (float): The apothem of the pentagonal base.
        height (float): The height of the prism.

    Returns:
        float: The total surface area.
    """
    base_area = 2.5 * side_length * base_apothem
    lateral_area = 5 * side_length * height
    return 2 * base_area + lateral_area


def calculate_pentagonal_prism_volume(
    side_length: float, base_apothem: float, height: float
) -> float:
    """Calculates the volume of a pentagonal prism.
    
    Note: The original formula `(5*b*h)/2` seems incorrect as volume
    is typically Base Area * Height. A standard formula is used here.

    Args:
        side_length (float): The length of a side of the pentagonal base.
        base_apothem (float): The apothem of the pentagonal base.
        height (float): The height of the prism.

    Returns:
        float: The volume of the prism.
    """
    base_area = 2.5 * side_length * base_apothem
    return base_area * height


def calculate_rectangular_wedge_volume(
    a: float, b: float, e: float, h: float
) -> float:
    """Calculates the volume of a rectangular right wedge.

    Args:
        a (float): Length of one parallel side of the triangular face.
        b (float): Width of the rectangular base.
        e (float): Length of the other parallel side of the triangular face.
        h (float): Height of the wedge.

    Returns:
        float: The volume of the wedge.
    """
    return ((b * h) / 6) * (2 * a + e)


def calculate_torus_volume(minor_radius: float, major_radius: float) -> float:
    """Calculates the volume of a torus.

    Args:
        minor_radius (float): The radius of the tube (r).
        major_radius (float): The radius from the center to the tube's center (R).

    Returns:
        float: The volume of the torus.
    """
    return 2 * (math.pi ** 2) * major_radius * (minor_radius ** 2)


def calculate_torus_surface_area(minor_radius: float, major_radius: float) -> float:
    """Calculates the surface area of a torus.

    Args:
        minor_radius (float): The radius of the tube (r).
        major_radius (float): The radius from the center to the tube's center (R).

    Returns:
        float: The surface area of the torus.
    """
    return 4 * (math.pi ** 2) * major_radius * minor_radius


def calculate_circle_area(radius: float) -> float:
    """Calculates the area of a circle given its radius.

    Args:
        radius (float): The radius of the circle.

    Returns:
        float: The area of the circle.
    """
    return math.pi * radius ** 2


def find_pythagorean_third_side(
    opposite: float | str = 'x', adjacent: float | str = 'x', hypotenuse: float | str = 'x'
) -> str:
    """Finds the missing side of a right-angled triangle.

    Provide two sides and mark the unknown side with 'x'.

    Args:
        opposite (float | str): The length of the opposite side. Defaults to 'x'.
        adjacent (float | str): The length of the adjacent side. Defaults to 'x'.
        hypotenuse (float | str): The length of the hypotenuse. Defaults to 'x'.

    Returns:
        str: A string describing the length of the missing side.
    """
    try:
        if hypotenuse == 'x':
            result = math.sqrt(opposite**2 + adjacent**2)
            return f"Hypotenuse = {result}"
        if opposite == 'x':
            result = math.sqrt(hypotenuse**2 - adjacent**2)
            return f"Opposite = {result}"
        if adjacent == 'x':
            result = math.sqrt(hypotenuse**2 - opposite**2)
            return f"Adjacent = {result}"
        return "All sides provided."
    except (ValueError, TypeError):
        return "Invalid input. Please provide two numeric values and one 'x'."


# =============================================================================
# 4. DATA STRUCTURE MANIPULATION
# =============================================================================

def convert_string_list_to_int_lists(data: list[str]) -> list[list[int]]:
    """Converts a list of string-represented lists into a list of integer lists.

    Example: ['[1, 2]', '[3, 4]'] -> [[1, 2], [3, 4]]

    Args:
        data (list[str]): A list where each element is a string like '[1, 2, 3]'.

    Returns:
        list[list[int]]: A list of lists containing integers.
    """
    return [ast.literal_eval(item) for item in data]


def convert_string_to_tuple_list(data: str) -> list[tuple[int, ...]]:
    """Converts a string of tuples into a list of actual tuples.

    Example: "(1, 2), (3, 4)" -> [(1, 2), (3, 4)]

    Args:
        data (str): A string containing comma-separated tuples.

    Returns:
        list[tuple[int, ...]]: A list of tuples.
    """
    # ast.literal_eval is the safest way to parse Python literals from a string.
    return ast.literal_eval(data)


def convert_list_to_tuple_of_single_item_lists(data: list) -> tuple[list, ...]:
    """Converts a list into a tuple where each element is a single-item list.

    Example: [1, 2, 3] -> ([1], [2], [3])

    Args:
        data (list): The input list.

    Returns:
        tuple[list, ...]: A tuple of single-item lists.
    """
    return tuple([item] for item in data)


def group_words_by_first_char(text: str) -> dict[str, list[str]]:
    """Creates a dictionary grouping words from text by their first character.

    Args:
        text (str): The input string to process.

    Returns:
        dict[str, list[str]]: A dictionary with first characters as keys
                                and lists of words as values.
    """
    word_groups = defaultdict(list)
    for word in text.split():
        first_char = word[0].lower()
        if word not in word_groups[first_char]:
            word_groups[first_char].append(word)
    return dict(word_groups)


def find_key_with_max_value(data: dict) -> any:
    """Finds the key in a dictionary corresponding to the maximum value.

    Args:
        data (dict): The input dictionary.

    Returns:
        any: The key associated with the highest value. Returns None if empty.
    """
    if not data:
        return None
    return max(data, key=data.get)


def get_next_key(data: dict, current_key: any) -> any:
    """Gets the next key in a dictionary based on insertion order.

    Args:
        data (dict): The input dictionary.
        current_key (any): The key from which to find the next one.

    Returns:
        any: The next key, or None if it's the last key or not found.
    """
    if current_key not in data:
        return None
    
    keys_list = list(data.keys())
    try:
        current_index = keys_list.index(current_key)
        if current_index + 1 < len(keys_list):
            return keys_list[current_index + 1]
        return None  # It's the last key
    except ValueError:
        return None # Should not happen due to the check above, but safe to have.

# =============================================================================
# 5. LIST AND SEQUENCE OPERATIONS
# =============================================================================

def are_lists_identical(list1: list, list2: list) -> bool:
    """Checks if two lists are identical (same elements in the same order).

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        bool: True if lists are identical, False otherwise.
    """
    return list1 == list2


def have_lists_same_elements(list1: list, list2: list) -> bool:
    """Checks if two lists have the same elements, regardless of order.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        bool: True if lists have the same elements, False otherwise.
    """
    return Counter(list1) == Counter(list2)


def calculate_median_of_nested_lists(data: list[tuple]) -> float:
    """Flattens a list of tuples/lists and calculates the median.

    Args:
        data (list[tuple]): A list of tuples or lists containing numbers.

    Returns:
        float: The median of all numbers.
    """
    flat_list = [item for sublist in data for item in sublist]
    flat_list.sort()
    
    mid_index = len(flat_list) // 2
    # For even length lists, median is the average of the two middle elements.
    # `~mid_index` is a bitwise way of getting `-(mid_index + 1)`, which
    # effectively gets the corresponding element from the end of the list.
    median = (flat_list[mid_index] + flat_list[~mid_index]) / 2
    return median


def multiply_all_elements_in_nested_list(data: list[tuple]) -> int | float:
    """Flattens a list of tuples/lists and multiplies all elements together.

    Args:
        data (list[tuple]): A list of tuples or lists of numbers.

    Returns:
        int | float: The product of all numbers.
    """
    flat_list = [item for sublist in data for item in sublist]
    # math.prod is available in Python 3.8+
    if hasattr(math, 'prod'):
        return math.prod(flat_list)
    
    # Manual implementation for older Python versions
    product = 1
    for item in flat_list:
        product *= item
    return product


def get_odd_product_pair(numbers: list[int]) -> tuple[int, int] | None:
    """Finds a distinct pair of numbers in a list whose product is odd.

    An odd product can only be formed by multiplying two odd numbers.

    Args:
        numbers (list[int]): A list of integers.

    Returns:
        tuple[int, int] | None: A tuple with the pair, or None if no such pair exists.
    """
    odd_numbers = [n for n in numbers if n % 2 != 0]
    if len(odd_numbers) >= 2:
        return (odd_numbers[0], odd_numbers[1])
    return None


def get_absent_digits(mobile_number: str) -> list[int]:
    """Finds digits (0-9) that are absent from a given number string.

    Args:
        mobile_number (str): A string representing the number.

    Returns:
        list[int]: A sorted list of absent digits.
    """
    present_digits = {int(digit) for digit in mobile_number}
    all_digits = set(range(10))
    absent = sorted(list(all_digits - present_digits))
    return absent


def generate_permutations(items: list) -> list[list]:
    """Generates all possible permutations of a list of distinct items.

    Args:
        items (list): A list of items to permute.

    Returns:
        list[list]: A list containing all permutations.
    """
    # This is a classic recursive approach.
    # For a simple solution, `itertools.permutations` is standard.
    # from itertools import permutations
    # return list(permutations(items))

    # Re-implementing the provided logic for completeness
    if not items:
        return [[]]
    
    first = items[0]
    rest = items[1:]
    
    perms_without_first = generate_permutations(rest)
    all_permutations = []
    
    for perm in perms_without_first:
        for i in range(len(perm) + 1):
            new_perm = perm[:i] + [first] + perm[i:]
            all_permutations.append(new_perm)
            
    return all_permutations


# =============================================================================
# 6. STRING OPERATIONS
# =============================================================================

def repeat_string(text: str, times: int) -> str:
    """Repeats a given string n times.

    Args:
        text (str): The string to repeat.
        times (int): A non-negative integer for the number of repetitions.

    Returns:
        str: The resulting repeated string.
    """
    if times < 0:
        raise ValueError("Number of repetitions cannot be negative.")
    return text * times


def repeat_substring(text: str, times: int, length: int = 2) -> str:
    """Repeats the first n characters of a string.

    If the string length is less than n, the whole string is used.

    Args:
        text (str): The input string.
        times (int): The number of repetitions.
        length (int): The length of the prefix to copy. Defaults to 2.

    Returns:
        str: The resulting string.
    """
    prefix = text[:length]
    return prefix * times


def reverse_string(text: str) -> str:
    """Reverses a string.

    Args:
        text (str): The string to reverse.

    Returns:
        str: The reversed string.
    """
    return text[::-1]


def is_palindrome(text: str) -> bool:
    """Checks if a string is a palindrome (reads the same forwards and backwards).

    Args:
        text (str): The string to check.

    Returns:
        bool: True if the string is a palindrome, False otherwise.
    """
    return text == text[::-1]


def count_upper_lower_case(text: str) -> dict[str, int]:
    """Counts the number of uppercase and lowercase letters in a string.

    Args:
        text (str): The input string.

    Returns:
        dict[str, int]: A dictionary with counts for 'UPPER_CASE' and 'LOWER_CASE'.
    """
    counts = {"UPPER_CASE": 0, "LOWER_CASE": 0}
    for char in text:
        if char.isupper():
            counts["UPPER_CASE"] += 1
        elif char.islower():
            counts["LOWER_CASE"] += 1
    return counts


# =============================================================================
# 7. NUMERIC AND MATH UTILITIES
# =============================================================================

def add_numbers(a: int | float, b: int | float) -> int | float:
    """Adds two numbers."""
    return a + b


def swap_variables(a: any, b: any) -> tuple[any, any]:
    """Swaps the values of two variables.

    Args:
        a (any): The first variable.
        b (any): The second variable.

    Returns:
        tuple[any, any]: A tuple with the swapped values (b, a).
    """
    return b, a


def is_near_thousand(n: int) -> bool:
    """Checks if a number is within 100 of 1000 or 2000.

    Args:
        n (int): The number to check.

    Returns:
        bool: True if the condition is met, False otherwise.
    """
    return (abs(1000 - n) <= 100) or (abs(2000 - n) <= 100)


def sum_thrice_if_equal(x: int, y: int, z: int) -> int:
    """Calculates the sum of three numbers.

    If all three numbers are equal, it returns thrice their sum.

    Args:
        x (int): First number.
        y (int): Second number.
        z (int): Third number.

    Returns:
        int: The sum, or thrice the sum if all numbers are equal.
    """
    total = x + y + z
    if x == y == z:
        return total * 3
    return total


def calculate_n_plus_nn_plus_nnn(n: int) -> int:
    """Computes the value of n + nn + nnn for a given integer n.

    Example: for n=5, calculates 5 + 55 + 555.

    Args:
        n (int): The base integer digit.

    Returns:
        int: The result of the computation.
    """
    s = str(n)
    n1 = int(s)
    n2 = int(s + s)
    n3 = int(s + s + s)
    return n1 + n2 + n3


def sum_of_cubes_smaller_than(n: int) -> int:
    """Returns the sum of the cubes of all positive integers smaller than n.

    Args:
        n (int): A positive integer.

    Returns:
        int: The sum of cubes.
    """
    if n <= 1:
        return 0
    # Using the formula for sum of cubes: (k * (k+1) / 2)^2 where k = n-1
    k = n - 1
    return int((k * (k + 1) / 2) ** 2)


def check_divisibility(dividend: int, divisor: int) -> bool:
    """Checks if a number is divisible by another.

    Args:
        dividend (int): The number to be divided.
        divisor (int): The number to divide by.

    Returns:
        bool: True if divisible, False otherwise.
    """
    if divisor == 0:
        raise ValueError("Cannot divide by zero.")
    return dividend % divisor == 0


def calculate_factorial(n: int) -> int:
    """Calculates the factorial of a non-negative integer using recursion.

    Args:
        n (int): A non-negative integer.

    Returns:
        int: The factorial of n.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0:
        return 1
    return n * calculate_factorial(n - 1)


def is_prime(n: int) -> bool:
    """Checks if a number is a prime number.

    Args:
        n (int): The number to check.

    Returns:
        bool: True if the number is prime, False otherwise.
    """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def find_primes_in_range(start: int, end: int) -> list[int]:
    """Finds all prime numbers within a given interval.

    Args:
        start (int): The start of the interval (inclusive).
        end (int): The end of the interval (inclusive).

    Returns:
        list[int]: A list of prime numbers found in the range.
    """
    return [num for num in range(start, end + 1) if is_prime(num)]


def is_armstrong_number(n: int) -> bool:
    """Checks if a number is an Armstrong number.

    An Armstrong number is a number that is equal to the sum of its own digits
    each raised to the power of the number of digits.

    Args:
        n (int): The number to check.

    Returns:
        bool: True if it is an Armstrong number, False otherwise.
    """
    if n < 0:
        return False
    s = str(n)
    order = len(s)
    
    total = sum(int(digit) ** order for digit in s)
    
    return n == total


def fibonacci_sequence(n_terms: int) -> list[int]:
    """Generates the Fibonacci sequence up to n terms.

    Args:
        n_terms (int): The number of terms to generate.

    Returns:
        list[int]: A list containing the Fibonacci sequence.
    """
    if n_terms <= 0:
        return []
    if n_terms == 1:
        return [0]
    
    sequence = [0, 1]
    while len(sequence) < n_terms:
        next_val = sequence[-1] + sequence[-2]
        sequence.append(next_val)
        
    return sequence


def get_nth_fibonacci(n: int) -> int:
    """Returns the nth element of the Fibonacci series (1-indexed).

    Args:
        n (int): The position in the series (e.g., n=1 is 0, n=2 is 1).

    Returns:
        int: The nth Fibonacci number.
    """
    if n <= 0:
        raise ValueError("Input must be a positive integer.")
    if n == 1:
        return 0
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return a


def is_leap_year(year: int) -> bool:
    """Determines if a year is a leap year.

    Args:
        year (int): The year to check.

    Returns:
        bool: True if it is a leap year, False otherwise.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


# =============================================================================
# 8. MISCELLANEOUS UTILITIES
# =============================================================================

def get_days_between_dates(date1_str: str, date2_str: str, fmt: str = "%d/%m/%Y") -> int:
    """Calculates the number of days between two dates.

    Args:
        date1_str (str): The start date string.
        date2_str (str): The end date string.
        fmt (str): The date format string. Defaults to "%d/%m/%Y".

    Returns:
        int: The absolute difference in days.
    """
    d1 = datetime.datetime.strptime(date1_str, fmt).date()
    d2 = datetime.datetime.strptime(date2_str, fmt).date()
    return abs((d2 - d1).days)


def print_histogram(items: list[int], symbol: str = '*'):
    """Prints a histogram from a list of integers.

    Args:
        items (list[int]): A list of integers representing bar lengths.
        symbol (str): The character to use for the bars. Defaults to '*'.
    """
    for n in items:
        print(symbol * n)


def kilometers_to_miles(km: float) -> float:
    """Converts kilometers to miles.

    Args:
        km (float): Distance in kilometers.

    Returns:
        float: Distance in miles.
    """
    return km * KILOMETERS_TO_MILES


def celsius_to_fahrenheit(celsius: float) -> float:
    """Converts temperature from Celsius to Fahrenheit.

    Args:
        celsius (float): Temperature in degrees Celsius.

    Returns:
        float: Temperature in degrees Fahrenheit.
    """
    return (celsius * 1.8) + 32


# =============================================================================
# 9. MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Main function to demonstrate the usage of all refactored functions."""
    print("--- Refactored Python Examples ---")

    # --- Geometry Demonstrations ---
    print("\n--- Geometry ---")
    print(f"Cylinder CSA (r=5, h=8): {calculate_cylinder_csa(5, 8):.2f}")
    print(f"Icosahedron Area (a=5): {calculate_icosahedron_area(5):.2f}")
    print(f"Icosahedron Volume (a=5): {calculate_icosahedron_volume(5):.2f}")
    print(f"Torus Volume (r=3, R=7): {calculate_torus_volume(3, 7):.2f}")
    print(f"Torus Surface Area (r=3, R=7): {calculate_torus_surface_area(3, 7):.2f}")
    print(f"Circle Area (r=5): {calculate_circle_area(5):.2f}")
    print(f"Pythagoras (3, 4, 'x'): {find_pythagorean_third_side(3, 4, 'x')}")

    # --- Data Structure Demonstrations ---
    print("\n--- Data Structures ---")
    str_list = ['[1, 4, 5]', '[4, 6, 8]']
    print(f"Convert '{str_list}' to int lists: {convert_string_list_to_int_lists(str_list)}")
    
    str_tuple = "(1, 3, 4), (5, 6, 4), (1, 3, 6)"
    print(f"Convert '{str_tuple}' to tuple list: {convert_string_to_tuple_list(str_tuple)}")
    
    text_to_group = "GeeksforGeeks is a Computer Science portal for geeks."
    print(f"Words grouped by first char: {group_words_by_first_char(text_to_group)}")

    ratings = {'BreakingBad': 100, 'GameOfThrones': 1292, 'TMKUC': 88}
    print(f"Key with max value in {ratings}: {find_key_with_max_value(ratings)}")
    
    test_dict = {'gfg': 1, 'is': 2, 'best': 3}
    print(f"Next key after 'is' in {test_dict}: {get_next_key(test_dict, 'is')}")

    # --- List/Sequence Demonstrations ---
    print("\n--- Lists and Sequences ---")
    list1, list2 = [1, 2, 3], [1, 2, 3]
    list3 = [3, 2, 1]
    print(f"Are {list1} and {list2} identical? {are_lists_identical(list1, list2)}")
    print(f"Are {list1} and {list3} identical? {are_lists_identical(list1, list3)}")
    print(f"Do {list1} and {list3} have the same elements? {have_lists_same_elements(list1, list3)}")

    nested_data = [(1, 4, 5), (7, 8), (2, 4, 10)]
    print(f"Median of {nested_data}: {calculate_median_of_nested_lists(nested_data)}")
    
    product_data = [(2, 4), (6, 7), (5, 1)]
    print(f"Product of {product_data}: {multiply_all_elements_in_nested_list(product_data)}")
    
    permutations = generate_permutations([1, 2, 3])
    print(f"Permutations of [1, 2, 3]: {permutations}")

    # --- String Demonstrations ---
    print("\n--- Strings ---")
    print(f"Repeat 'abc' 3 times: {repeat_string('abc', 3)}")
    print(f"Reverse '1234abcd': {reverse_string('1234abcd')}")
    print(f"Is 'aza' a palindrome? {is_palindrome('aza')}")
    print(f"Case count for 'The quick Brown Fox': {count_upper_lower_case('The quick Brown Fox')}")

    # --- Numeric/Math Demonstrations ---
    print("\n--- Numeric and Math ---")
    x, y = 5, 10
    swapped_x, swapped_y = swap_variables(x, y)
    print(f"Swapping {x} and {y} gives: {swapped_x}, {swapped_y}")
    
    print(f"Is 950 near 1000 or 2000? {is_near_thousand(950)}")
    print(f"Sum thrice if equal (1, 2, 3): {sum_thrice_if_equal(1, 2, 3)}")
    print(f"Sum thrice if equal (3, 3, 3): {sum_thrice_if_equal(3, 3, 3)}")
    print(f"n+nn+nnn for n=5: {calculate_n_plus_nn_plus_nnn(5)}")
    
    print(f"Factorial of 7: {calculate_factorial(7)}")
    print(f"Is 29 a prime number? {is_prime(29)}")
    print(f"Primes between 10 and 50: {find_primes_in_range(10, 50)}")
    print(f"Is 153 an Armstrong number? {is_armstrong_number(153)}")
    print(f"Is 154 an Armstrong number? {is_armstrong_number(154)}")
    
    print(f"First 10 Fibonacci numbers: {fibonacci_sequence(10)}")
    print(f"The 10th Fibonacci number is: {get_nth_fibonacci(10)}")

    print(f"Is 2024 a leap year? {is_leap_year(2024)}")
    print(f"Is 2023 a leap year? {is_leap_year(2023)}")

    # --- Miscellaneous Demonstrations ---
    print("\n--- Miscellaneous ---")
    print(f"Days between 01/01/2023 and 31/12/2023: {get_days_between_dates('01/01/2023', '31/12/2023')}")
    print("Histogram for [2, 3, 6, 5]:")
    print_histogram([2, 3, 6, 5])
    
    print(f"5 km to miles: {kilometers_to_miles(5.0):.2f}")
    print(f"37.5 C to Fahrenheit: {celsius_to_fahrenheit(37.5):.1f}")
    
    mobile = "9832209763"
    print(f"Absent digits in {mobile}: {get_absent_digits(mobile)}")

if __name__ == "__main__":
    main()