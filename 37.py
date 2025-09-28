# -*- coding: utf-8 -*-
"""
A collection of refactored Python code examples.

This script demonstrates Python standard practices by taking simple, procedural code
snippets and refactoring them into well-structured, documented, and readable functions.

Key improvements include:
- Encapsulation in functions
- Descriptive naming for variables and functions
- Docstrings and type hints
- Use of Pythonic idioms (e.g., comprehensions)
- Separation of calculation and presentation (I/O)
- A main execution block for demonstration purposes.
"""

# -----------------------------------------------------------------------------
# 1. Imports - Grouped at the top of the file
# -----------------------------------------------------------------------------
import itertools
import math
import re
import random
import string
from datetime import date, datetime, timedelta
from typing import List, Tuple, Dict, Any, Generator, Optional

# -----------------------------------------------------------------------------
# 2. String Manipulation
# -----------------------------------------------------------------------------

def get_even_index_chars(input_str: str) -> str:
    """Returns characters from a string that have even indices.

    Args:
        input_str: The string to process.

    Returns:
        A new string containing characters from even-numbered indices.
    """
    return input_str[::2]


def sort_words_alphabetically(comma_separated_words: str) -> str:
    """Sorts a comma-separated string of words alphabetically.

    Args:
        comma_separated_words: A string of words separated by commas.

    Returns:
        A comma-separated string of the sorted words.
    """
    words = [word.strip() for word in comma_separated_words.split(',')]
    words.sort()
    return ','.join(words)


def process_and_capitalize_lines(lines: List[str]) -> List[str]:
    """Capitalizes a list of strings.

    Args:
        lines: A list of strings.

    Returns:
        A new list with all strings converted to uppercase.
    """
    return [line.upper() for line in lines]


def sort_unique_words(whitespace_separated_words: str) -> str:
    """Removes duplicate words and sorts them alphanumerically.

    Args:
        whitespace_separated_words: A string of words separated by whitespace.

    Returns:
        A whitespace-separated string of unique, sorted words.
    """
    words = whitespace_separated_words.split(' ')
    unique_sorted_words = sorted(list(set(words)))
    return ' '.join(unique_sorted_words)


def count_letters_and_digits(sentence: str) -> Dict[str, int]:
    """Counts the number of letters and digits in a sentence.

    Args:
        sentence: The input string.

    Returns:
        A dictionary with counts for "LETTERS" and "DIGITS".
    """
    counts = {"LETTERS": 0, "DIGITS": 0}
    for char in sentence:
        if char.isalpha():
            counts["LETTERS"] += 1
        elif char.isdigit():
            counts["DIGITS"] += 1
    return counts


def count_upper_and_lower_case(sentence: str) -> Dict[str, int]:
    """Counts the number of uppercase and lowercase letters in a sentence.

    Args:
        sentence: The input string.

    Returns:
        A dictionary with counts for "UPPER CASE" and "LOWER CASE".
    """
    counts = {"UPPER CASE": 0, "LOWER CASE": 0}
    for char in sentence:
        if char.isupper():
            counts["UPPER CASE"] += 1
        elif char.islower():
            counts["LOWER CASE"] += 1
    return counts


def is_palindrome(text: str) -> bool:
    """Checks if a given string is a palindrome.

    Args:
        text: The string to check.

    Returns:
        True if the string is a palindrome, False otherwise.
    """
    # A more Pythonic way is to use slicing to reverse the string
    return text == text[::-1]


def to_camel_case(snake_case_str: str) -> str:
    """Converts a snake_case or kebab-case string to camelCase.

    Args:
        snake_case_str: The input string.

    Returns:
        The camelCase version of the string.
    """
    s = re.sub(r"(_|-)+", " ", snake_case_str).title().replace(" ", "")
    return s[0].lower() + s[1:] if s else ""


def capitalize_first_letter(text: str, lower_rest: bool = False) -> str:
    """Capitalizes the first letter of a string.

    Args:
        text: The string to capitalize.
        lower_rest: If True, converts the rest of the string to lowercase.

    Returns:
        The capitalized string.
    """
    if not text:
        return ""
    first_char = text[0].upper()
    rest = text[1:].lower() if lower_rest else text[1:]
    return f"{first_char}{rest}"


def remove_vowels(text: str) -> str:
    """Removes all vowels from a string.

    Args:
        text: The input string.

    Returns:
        The string with vowels removed.
    """
    vowels = "aeiouAEIOU"
    return "".join(char for char in text if char not in vowels)


def remove_punctuation(text: str) -> str:
    """Removes all punctuation from a string.

    Args:
        text: The input string.

    Returns:
        The string without punctuation.
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def get_byte_size(text: str) -> int:
    """Returns the length of a string in bytes (UTF-8 encoded).

    Args:
        text: The string to measure.

    Returns:
        The size of the string in bytes.
    """
    return len(text.encode('utf-8'))


# -----------------------------------------------------------------------------
# 3. Number and Math Operations
# -----------------------------------------------------------------------------

def solve_chicken_rabbit_puzzle(num_heads: int, num_legs: int) -> Optional[Tuple[int, int]]:
    """Solves the classic chicken and rabbit puzzle.

    Given a total number of heads and legs, finds the number of chickens and rabbits.

    Args:
        num_heads: Total number of heads.
        num_legs: Total number of legs.

    Returns:
        A tuple of (num_chickens, num_rabbits) or None if no solution exists.
    """
    for num_chickens in range(num_heads + 1):
        num_rabbits = num_heads - num_chickens
        if (2 * num_chickens) + (4 * num_rabbits) == num_legs:
            return num_chickens, num_rabbits
    return None


def find_divisible_by_7_not_5(start: int, end: int) -> List[int]:
    """Finds numbers in a range divisible by 7 but not by 5.

    Args:
        start: The starting number of the range (inclusive).
        end: The ending number of the range (inclusive).

    Returns:
        A list of numbers that meet the criteria.
    """
    return [num for num in range(start, end + 1) if num % 7 == 0 and num % 5 != 0]


def factorial(n: int) -> int:
    """Computes the factorial of a number using recursion.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial of n.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    return n * factorial(n - 1)


def calculate_formula_q(d_values: List[float]) -> List[int]:
    """Calculates Q = sqrt((2 * C * D) / H) for a list of D values.

    Constants C and H are fixed at 50 and 30, respectively.

    Args:
        d_values: A list of D values.

    Returns:
        A list of calculated and rounded Q values.
    """
    CONSTANT_C = 50
    CONSTANT_H = 30
    results = []
    for d in d_values:
        q = math.sqrt((2 * CONSTANT_C * d) / CONSTANT_H)
        results.append(round(q))
    return results


def compute_a_plus_aa(digit_char: str) -> int:
    """Computes the value of a + aa + aaa + aaaa for a given digit 'a'.

    Args:
        digit_char: A single character string representing a digit (e.g., '9').

    Returns:
        The integer sum.
    """
    # More general and robust approach
    n1 = int(digit_char)
    n2 = int(digit_char * 2)
    n3 = int(digit_char * 3)
    n4 = int(digit_char * 4)
    return n1 + n2 + n3 + n4


def filter_even_digit_numbers(start: int, end: int) -> List[int]:
    """Finds numbers where every digit is even.

    Args:
        start: The starting number of the range (inclusive).
        end: The ending number of the range (inclusive).

    Returns:
        A list of numbers where all digits are even.
    """
    result = []
    for num in range(start, end + 1):
        if all(int(digit) % 2 == 0 for digit in str(num)):
            result.append(num)
    return result


def celsius_to_fahrenheit(degrees_celsius: float) -> float:
    """Converts temperature from Celsius to Fahrenheit."""
    return (degrees_celsius * 1.8) + 32


def generate_otp(length: int = 6) -> str:
    """Generates a numeric OTP of a specified length.

    Args:
        length: The desired length of the OTP.

    Returns:
        A string representing the OTP.
    """
    return "".join(random.choices(string.digits, k=length))


def greatest_common_divisor(a: int, b: int) -> int:
    """Calculates the greatest common divisor (GCD) of two integers."""
    return math.gcd(a, b)


def least_common_multiple(a: int, b: int) -> int:
    """Calculates the least common multiple (LCM) of two integers."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)
    
    
# -----------------------------------------------------------------------------
# 4. List, Tuple, and Dictionary Operations
# -----------------------------------------------------------------------------

def generate_permutations(items: List[Any]) -> List[Tuple[Any, ...]]:
    """Generates all permutations of a list.

    Args:
        items: The list of items to permute.

    Returns:
        A list of tuples, where each tuple is a permutation.
    """
    return list(itertools.permutations(items))


def generate_squares_dict(n: int) -> Dict[int, int]:
    """Generates a dictionary of (i, i*i) for i from 1 to n.

    Args:
        n: The upper limit (inclusive).

    Returns:
        A dictionary with numbers as keys and their squares as values.
    """
    return {i: i * i for i in range(1, n + 1)}


def create_list_and_tuple_from_string(comma_separated_numbers: str) -> Tuple[List[str], Tuple[str, ...]]:
    """Creates a list and a tuple from a comma-separated string of numbers.

    Args:
        comma_separated_numbers: The input string.

    Returns:
        A tuple containing the generated list and tuple.
    """
    items = [item.strip() for item in comma_separated_numbers.split(',')]
    return items, tuple(items)


def create_multiplication_matrix(rows: int, cols: int) -> List[List[int]]:
    """Generates a 2D array where element [i, j] is i * j.

    Args:
        rows: The number of rows.
        cols: The number of columns.

    Returns:
        A 2D list representing the multiplication matrix.
    """
    return [[row * col for col in range(cols)] for row in range(rows)]


def filter_binary_divisible_by_5(binary_numbers: List[str]) -> List[str]:
    """Filters a list of binary strings, returning those divisible by 5.

    Args:
        binary_numbers: A list of strings, each representing a binary number.

    Returns:
        A list of binary strings that are divisible by 5.
    """
    return [b for b in binary_numbers if int(b, 2) % 5 == 0]


def get_squared_odd_numbers(numbers: List[int]) -> List[int]:
    """Squares each odd number in a list.

    Args:
        numbers: A list of integers.

    Returns:
        A list containing the squares of only the odd numbers.
    """
    return [x * x for x in numbers if x % 2 != 0]


def numbers_divisible_by_n_generator(n: int, divisor: int) -> Generator[int, None, None]:
    """A generator for numbers divisible by a given divisor, up to n.

    Args:
        n: The upper limit (exclusive).
        divisor: The number to divide by.

    Yields:
        Numbers from 0 to n-1 that are divisible by the divisor.
    """
    for i in range(n):
        if i % divisor == 0:
            yield i


def count_word_frequencies(text: str) -> Dict[str, int]:
    """Counts the frequency of words in a text.

    Args:
        text: The input string.

    Returns:
        A dictionary mapping words to their frequencies.
    """
    freq = {}
    for word in text.split():
        freq[word] = freq.get(word, 0) + 1
    return freq


def are_all_elements_equal(items: List[Any]) -> bool:
    """Checks if all elements in a list are equal."""
    if not items:
        return True
    return len(set(items)) == 1


def are_all_elements_unique(items: List[Any]) -> bool:
    """Checks if all elements in a list are unique."""
    return len(items) == len(set(items))


def find_most_frequent_element(items: List[Any]) -> Optional[Any]:
    """Finds the most frequent element in a list."""
    if not items:
        return None
    return max(set(items), key=items.count)
    
# -----------------------------------------------------------------------------
# 5. Classes and Object-Oriented Programming
# -----------------------------------------------------------------------------

class InputOutString:
    """A class to get a string from stdin and print its uppercase version."""
    def __init__(self):
        self.user_string = ""

    def get_string(self):
        """Reads a string from standard input."""
        self.user_string = input("Enter a string: ")

    def print_string(self):
        """Prints the stored string in uppercase."""
        print(self.user_string.upper())


class Person:
    """Demonstrates class vs. instance attributes."""
    # Class attribute - shared by all instances of the class
    species = "Homo sapiens"

    def __init__(self, name: str):
        # Instance attribute - unique to each instance
        self.name = name
        
# -----------------------------------------------------------------------------
# 6. File and System Operations
# -----------------------------------------------------------------------------

def get_builtin_docstring(func_name: str) -> str:
    """Retrieves the docstring of a Python built-in function."""
    try:
        # Using eval is generally risky, but here it's controlled.
        # A safer approach might use a dictionary mapping strings to functions.
        return eval(f"{func_name}.__doc__")
    except (NameError, AttributeError):
        return f"Could not find docstring for '{func_name}'."


def get_image_resolution(filename: str) -> Optional[Tuple[int, int]]:
    """Reads the resolution of a JPEG image file.

    Args:
        filename: The path to the JPEG file.

    Returns:
        A tuple of (width, height) or None if an error occurs.
    """
    try:
        with open(filename, 'rb') as img_file:
            # This is a simplified example; a robust solution would parse JPEG markers.
            img_file.seek(163)
            a = img_file.read(2)
            height = (a[0] << 8) + a[1]
            a = img_file.read(2)
            width = (a[0] << 8) + a[1]
        return width, height
    except (IOError, IndexError):
        return None
        
# -----------------------------------------------------------------------------
# 7. Main Demonstration Block
# -----------------------------------------------------------------------------

def main():
    """Main function to demonstrate the refactored code."""
    print("--- Refactored Python Examples ---")

    # String Manipulation
    print("\n1. Characters at even indices in 'HelloWorld':")
    print(f"   Result: {get_even_index_chars('HelloWorld')}")

    print("\n2. Sorting 'red,white,black,blue' alphabetically:")
    print(f"   Result: {sort_words_alphabetically('red,white,black,blue')}")

    # Number and Math
    print("\n3. Solving the chicken and rabbit puzzle (35 heads, 94 legs):")
    solution = solve_chicken_rabbit_puzzle(35, 94)
    if solution:
        chickens, rabbits = solution
        print(f"   Result: {chickens} chickens and {rabbits} rabbits.")
    else:
        print("   No solution found.")
        
    print("\n4. Numbers between 2000-3200 divisible by 7 but not 5:")
    div_nums = find_divisible_by_7_not_5(2000, 3200)
    print(f"   Found {len(div_nums)} numbers. First 5: {div_nums[:5]}")

    print("\n5. Factorial of 8:")
    print(f"   Result: {factorial(8)}")

    # List, Tuple, Dictionary
    print("\n6. Generating permutations of [1, 2, 3]:")
    print(f"   Result: {generate_permutations([1, 2, 3])}")

    print("\n7. Generating a dictionary of squares up to 8:")
    print(f"   Result: {generate_squares_dict(8)}")
    
    print("\n8. Create multiplication matrix (3x4):")
    print(f"   Result: {create_multiplication_matrix(3, 4)}")
    
    # Class Demonstration
    print("\n9. Class Demonstration (InputOutString):")
    # str_obj = InputOutString()
    # str_obj.get_string() # This would pause the script for input
    # str_obj.print_string()
    print("   (Skipping interactive input in demo)")
    
    print("\n10. Class vs. Instance attribute demo:")
    person1 = Person("Alice")
    person2 = Person("Bob")
    print(f"   {person1.name}'s species: {person1.species}")
    print(f"   {person2.name}'s species: {person2.species}")
    print(f"   Changing class attribute Person.species to 'Homo neanderthalensis'")
    Person.species = "Homo neanderthalensis"
    print(f"   {person1.name}'s new species: {person1.species}")

    # Built-in Docs
    print("\n11. Docstring for abs():")
    print(get_builtin_docstring('abs'))


if __name__ == "__main__":
    main()