# datetime_utils.py

"""
A collection of utility functions for date and time operations.
"""
from datetime import date, datetime


def get_days_between_dates(year: int, month: int, day: int) -> int:
    """
    Calculates the number of days between a given date and today.

    Args:
        year: The year of the date.
        month: The month of the date.
        day: The day of the date.

    Returns:
        The difference in days as an integer.
    """
    past_date = date(year, month, day)
    today_date = date.today()
    delta = today_date - past_date
    return delta.days


def get_current_datetime_str() -> str:
    """
    Gets the current date and time as a formatted string.

    Returns:
        A string representing the current date and time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# math_operations.py

"""
A collection of common mathematical and numerical functions.
"""
import math
import cmath  # For complex numbers
from typing import List, Union

Numeric = Union[int, float]


def is_leap_year(year: int) -> bool:
    """
    Checks if a given year is a leap year.

    A year is a leap year if it is divisible by 4, except for end-of-century
    years, which must be divisible by 400.

    Args:
        year: The year to check.

    Returns:
        True if the year is a leap year, False otherwise.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def calculate_mean(numbers: List[Numeric]) -> float:
    """
    Calculates the arithmetic mean of a list of numbers.

    Args:
        numbers: A list of integers or floats.

    Returns:
        The mean of the numbers as a float. Returns 0 for an empty list.
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def is_prime(number: int) -> bool:
    """
    Checks if a number is a prime number using an optimized algorithm.

    Args:
        number: The integer to check.

    Returns:
        True if the number is prime, False otherwise.
    """
    if number < 2:
        return False
    # Check for divisibility only up to the square root of the number.
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False
    return True


def is_perfect_square(number: int) -> bool:
    """
    Checks if a number is a perfect square.

    Args:
        number: The integer to check.

    Returns:
        True if the number is a perfect square, False otherwise.
    """
    if number < 0:
        return False
    sqrt_num = int(math.sqrt(number))
    return sqrt_num * sqrt_num == number


def get_hypotenuse(base: Numeric, height: Numeric) -> float:
    """Calculates the hypotenuse of a right-angled triangle."""
    return math.hypot(base, height)  # More accurate than sqrt(a**2 + b**2)


def sum_digits(number: int) -> int:
    """Calculates the sum of the digits of a non-negative integer."""
    total = 0
    num_str = str(abs(number))
    for digit in num_str:
        total += int(digit)
    return total


def calculate_bmi(height_m: float, weight_kg: float) -> float:
    """
    Calculates Body Mass Index (BMI).

    Args:
        height_m: Height in meters.
        weight_kg: Weight in kilograms.

    Returns:
        The calculated BMI. Raises ValueError if height is zero.
    """
    if height_m <= 0:
        raise ValueError("Height must be positive.")
    return weight_kg / (height_m ** 2)


def get_fibonacci_sequence(n: int) -> List[int]:
    """Generates the Fibonacci sequence up to the nth term (iterative)."""
    if n <= 0:
        return []
    sequence = [0, 1]
    if n == 1:
        return [0]
    while len(sequence) < n:
        next_val = sequence[-1] + sequence[-2]
        sequence.append(next_val)
    return sequence[:n]


def get_factors(number: int) -> List[int]:
    """Finds all factors of a given integer."""
    if number == 0:
        return []
    factors = []
    for i in range(1, int(math.sqrt(abs(number))) + 1):
        if number % i == 0:
            factors.append(i)
            if i*i != number:
                factors.append(abs(number) // i)
    factors.sort()
    return factors
    
def find_lcm(x: int, y: int) -> int:
    """Computes the least common multiple (LCM) of two integers."""
    if x == 0 or y == 0:
        return 0
    # Formula: LCM(a, b) = |a * b| / GCD(a, b)
    return abs(x * y) // math.gcd(x, y)


def find_gcd(x: int, y: int) -> int:
    """Computes the greatest common divisor (GCD) of two integers."""
    return math.gcd(x, y)


# string_manipulation.py

"""
A collection of utility functions for string manipulation and processing.
"""
import re
from collections import Counter
import string as string_constants

def replace_vowels(text: str, replacement_char: str) -> str:
    """
    Replaces all vowels in a string with a specified character.

    Args:
        text: The input string.
        replacement_char: The character to replace vowels with.

    Returns:
        The modified string.
    """
    vowels = "AEIOUaeiou"
    for vowel in vowels:
        text = text.replace(vowel, replacement_char)
    return text


def find_urls(text: str) -> list[str]:
    """Finds all URLs in a given string using regex."""
    # A common and robust regex for finding URLs
    url_regex = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_regex, text)


def count_vowels(sentence: str) -> int:
    """Counts the number of vowels in a sentence."""
    return sum(1 for char in sentence if char.lower() in "aeiou")


def is_palindrome(text: str) -> bool:
    """
    Checks if a string is a palindrome (reads the same forwards and backwards).
    Case-insensitive and ignores non-alphanumeric characters.
    """
    # Sanitize the string
    normalized_str = ''.join(char.lower() for char in text if char.isalnum())
    return normalized_str == normalized_str[::-1]


def count_character_frequency(text: str) -> dict[str, int]:
    """Counts the frequency of each character in a string."""
    return Counter(text)


def remove_punctuation(text: str) -> str:
    """Removes all punctuation from a string."""
    return text.translate(str.maketrans('', '', string_constants.punctuation))


# list_operations.py

"""
A collection of functions for common list operations.
"""
from typing import List, Any, TypeVar

T = TypeVar('T')  # Generic type for functions that work on any type

def remove_duplicates(items: List[T]) -> List[T]:
    """
    Removes duplicate elements from a list while preserving order.

    Args:
        items: A list with potential duplicate elements.

    Returns:
        A new list with duplicates removed.
    """
    # For hashable types, dict.fromkeys is a fast way to get unique items
    return list(dict.fromkeys(items))


def sum_even_numbers(numbers: List[int]) -> int:
    """Calculates the sum of all even numbers in a list."""
    return sum(num for num in numbers if num % 2 == 0)


def filter_even_numbers(numbers: List[int]) -> List[int]:
    """Returns a new list containing only the even numbers."""
    return [num for num in numbers if num % 2 == 0]


def filter_odd_numbers(numbers: List[int]) -> List[int]:
    """Returns a new list containing only the odd numbers."""
    return [num for num in numbers if num % 2 == 1]


def get_elementwise_sum(list1: List[float], list2: List[float]) -> List[float]:
    """
    Adds corresponding elements of two lists.
    
    Raises:
        ValueError: If lists have different lengths.
    """
    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length.")
    return [x + y for x, y in zip(list1, list2)]


# file_operations.py

"""
Utility functions for file system operations.
"""
import os

def write_string_to_file(filepath: str, content: str) -> None:
    """
    Writes a string to a file, overwriting it if it exists.

    Args:
        filepath: The path to the file.
        content: The string content to write.
    """
    try:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
    except IOError as e:
        print(f"Error writing to file {filepath}: {e}")


def count_lines_in_file(filepath: str) -> int:
    """
    Counts the number of lines in a text file.

    Returns:
        The number of lines, or 0 if the file cannot be read.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f)
    except IOError as e:
        print(f"Error reading file {filepath}: {e}")
        return 0


def get_filename_and_extension(filepath: str) -> tuple[str, str]:
    """
    Splits a file path into its name and extension.

    Returns:
        A tuple containing (filename, extension).
    """
    base = os.path.basename(filepath)
    return os.path.splitext(base)


# utility.py

"""
Miscellaneous utility classes and functions, including converters.
"""

class ComplexNumber:
    """Represents a complex number with real and imaginary parts."""

    def __init__(self, real: float, imaginary: float):
        self.real = real
        self.imaginary = imaginary

    def __repr__(self) -> str:
        """Provides a developer-friendly string representation."""
        return f"ComplexNumber(real={self.real}, imaginary={self.imaginary})"

    def __str__(self) -> str:
        """Provides a user-friendly string representation."""
        return f"{self.real} + {self.imaginary}j"


def feet_inches_to_cm(feet: float, inches: float) -> float:
    """Converts a height in feet and inches to centimeters."""
    total_inches = (feet * 12) + inches
    return total_inches * 2.54


def celsius_to_kelvin(celsius: float) -> float:
    """Converts temperature from Celsius to Kelvin."""
    return celsius + 273.15  # Using more precise value


def kgs_to_pounds(kgs: float) -> float:
    """Converts weight from kilograms to pounds."""
    return kgs * 2.20462


def miles_to_kms(miles: float) -> float:
    """Converts distance from miles to kilometers."""
    return miles * 1.60934


# main.py

"""
Demonstration script for the python-examples project.
This file shows how to use the functions from the various utility modules.
"""

import datetime_utils
import file_operations
import list_operations
import math_operations
import string_manipulation
import utility

def demonstrate_datetime():
    """Shows examples from the datetime_utils module."""
    print("\n--- DateTime Utilities ---")
    days_diff = datetime_utils.get_days_between_dates(2022, 1, 1)
    print(f"Days since Jan 1, 2022: {days_diff}")
    print(f"Current time: {datetime_utils.get_current_datetime_str()}")


def demonstrate_math():
    """Shows examples from the math_operations module."""
    print("\n--- Math Operations ---")
    print(f"Is 2024 a leap year? {math_operations.is_leap_year(2024)}")
    print(f"Is 2023 a leap year? {math_operations.is_leap_year(2023)}")
    numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Mean of {numbers}: {math_operations.calculate_mean(numbers)}")
    print(f"Is 17 prime? {math_operations.is_prime(17)}")
    print(f"Is 18 prime? {math_operations.is_prime(18)}")
    print(f"Factors of 48: {math_operations.get_factors(48)}")


def demonstrate_strings():
    """Shows examples from the string_manipulation module."""
    print("\n--- String Manipulation ---")
    original_str = "Hello World, this is a test!"
    replaced = string_manipulation.replace_vowels(original_str, "*")
    print(f"String with vowels replaced: '{replaced}'")
    
    palindrome_test = "A man, a plan, a canal: Panama"
    print(f"Is '{palindrome_test}' a palindrome? {string_manipulation.is_palindrome(palindrome_test)}")
    
    text_with_urls = "Check out https://www.python.org and http://example.com"
    print(f"URLs found: {string_manipulation.find_urls(text_with_urls)}")


def demonstrate_lists():
    """Shows examples from the list_operations module."""
    print("\n--- List Operations ---")
    my_list = [1, 2, 2, 3, 4, 4, 5, 6, 1]
    print(f"Original list: {my_list}")
    print(f"List with duplicates removed: {list_operations.remove_duplicates(my_list)}")
    
    numbers = [1, 2, 3, 4, 5, 6]
    print(f"Sum of even numbers in {numbers}: {list_operations.sum_even_numbers(numbers)}")
    print(f"Even numbers in {numbers}: {list_operations.filter_even_numbers(numbers)}")


def demonstrate_files():
    """Shows examples of file operations."""
    print("\n--- File Operations ---")
    filepath = "example.txt"
    content = "This is line one.\nThis is line two."
    file_operations.write_string_to_file(filepath, content)
    print(f"Wrote to '{filepath}'.")
    
    line_count = file_operations.count_lines_in_file(filepath)
    print(f"'{filepath}' has {line_count} lines.")
    
    name, ext = file_operations.get_filename_and_extension("mydocument.pdf")
    print(f"Filename: {name}, Extension: {ext}")


def main():
    """Main function to run all demonstrations."""
    print("Running Python Code Examples Demonstrations")
    demonstrate_datetime()
    demonstrate_math()
    demonstrate_strings()
    demonstrate_lists()
    demonstrate_files()


if __name__ == "__main__":
    main()

