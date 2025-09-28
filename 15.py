"""
A collection of simple Python code examples, refactored for clarity,
readability, and adherence to standard practices.
"""

# =============================================================================
# 1. Standard Library Imports
# =============================================================================
import argparse
import cmath  # For complex number square root, not in original but good practice
import csv
import datetime
import glob
import itertools
import logging
import math
import os
import random
import re
import time
from typing import Any, Callable, List, Tuple, Union

# Define a type alias for matrices for cleaner type hints
Matrix = List[List[Union[int, float]]]


# =============================================================================
# 2. Mathematical and Algorithmic Functions
# =============================================================================

def gcd(a: int, b: int) -> int:
    """
    Computes the Greatest Common Divisor (GCD/HCF) of two integers
    using the Euclidean algorithm.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The greatest common divisor of a and b.
    """
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    """
    Computes the Least Common Multiple (LCM) of two integers using the formula:
    LCM(a, b) = |a * b| / GCD(a, b).

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The least common multiple of a and b. Returns 0 if a or b is 0.
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)

def get_factors(number: int) -> List[int]:
    """
    Finds all factors of a given positive integer.

    Args:
        number: The integer to factor.

    Returns:
        A list of factors.
    """
    if number <= 0:
        return []
    factors = []
    for i in range(1, number + 1):
        if number % i == 0:
            factors.append(i)
    return factors

def fibonacci_recursive(n: int) -> int:
    """
    Calculates the nth Fibonacci number using recursion.
    Note: This is computationally expensive for large n.

    Args:
        n: The term in the sequence to find (0-indexed).

    Returns:
        The nth Fibonacci number.
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)

def sum_natural_recursive(n: int) -> int:
    """
    Calculates the sum of natural numbers up to n using recursion.

    Args:
        n: A positive integer.

    Returns:
        The sum of natural numbers from 1 to n.
    """
    if n <= 1:
        return n
    return n + sum_natural_recursive(n - 1)

def factorial_recursive(n: int) -> int:
    """
    Calculates the factorial of a non-negative integer using recursion.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial of n.
    
    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0:
        return 1
    if n == 1:
        return 1
    return n * factorial_recursive(n - 1)

def is_prime(num: int) -> bool:
    """Checks if a number is a prime number."""
    if num <= 1:
        return False
    # Check for factors from 2 up to the square root of num
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

# =============================================================================
# 3. Simple Calculator
# =============================================================================

def add(a: float, b: float) -> float:
    """Returns the sum of two numbers."""
    return a + b

def subtract(a: float, b: float) -> float:
    """Returns the difference of two numbers."""
    return a - b

def multiply(a: float, b: float) -> float:
    """Returns the product of two numbers."""
    return a * b

def divide(a: float, b: float) -> float:
    """
    Returns the quotient of two numbers.
    Raises ValueError on division by zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

def run_simple_calculator():
    """Runs an interactive command-line calculator."""
    print("\n--- Simple Calculator ---")
    print("Select operation:")
    print("1. Add")
    print("2. Subtract")
    print("3. Multiply")
    print("4. Divide")

    operations = {
        '1': ('+', add),
        '2': ('-', subtract),
        '3': ('*', multiply),
        '4': ('/', divide),
    }

    while True:
        choice = input("Enter choice (1/2/3/4) or 'q' to quit: ")
        if choice.lower() == 'q':
            break

        if choice in operations:
            try:
                num1 = float(input("Enter first number: "))
                num2 = float(input("Enter second number: "))
                
                op_symbol, op_func = operations[choice]
                result = op_func(num1, num2)
                
                print(f"Result: {num1} {op_symbol} {num2} = {result}")

            except ValueError as e:
                print(f"Invalid input: {e}. Please enter numbers.")
            except Exception as e:
                print(f"An error occurred: {e}")
            break
        else:
            print("Invalid Input. Please enter a valid choice.")


# =============================================================================
# 4. Matrix Operations
# =============================================================================

def add_matrices(X: Matrix, Y: Matrix) -> Matrix:
    """Adds two matrices of the same dimensions."""
    return [[X[i][j] + Y[i][j] for j in range(len(X[0]))] for i in range(len(X))]

def transpose_matrix(X: Matrix) -> Matrix:
    """Transposes a matrix."""
    return [[X[j][i] for j in range(len(X))] for i in range(len(X[0]))]

def multiply_matrices(X: Matrix, Y: Matrix) -> Matrix:
    """Multiplies two matrices using list comprehension."""
    return [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*Y)] for X_row in X]


# =============================================================================
# 5. String Manipulation
# =============================================================================

def is_palindrome(text: str) -> bool:
    """
    Checks if a string is a palindrome (caseless).
    """
    processed_text = text.casefold()
    return processed_text == processed_text[::-1]

def remove_punctuation(text: str, custom_punctuation: str = None) -> str:
    """
    Removes punctuation from a string.
    
    Args:
        text: The input string.
        custom_punctuation: Optional string of characters to remove.
                            Defaults to a standard set.
    
    Returns:
        The string with punctuation removed.
    """
    if custom_punctuation is None:
        # Standard punctuation set
        punctuation = '''!()-[]{};:'"\\,<>./?@#$%^&*_~'''
    else:
        punctuation = custom_punctuation

    # More efficient than repeated string concatenation
    return "".join(char for char in text if char not in punctuation)

def sort_words_alphabetically(sentence: str) -> List[str]:
    """
    Sorts the words in a sentence alphabetically (caseless).
    """
    words = [word.lower() for word in sentence.split()]
    words.sort()
    return words

def count_vowels(text: str) -> dict[str, int]:
    """
    Counts the occurrence of each vowel in a string (caseless).
    """
    text_lower = text.casefold()
    vowels = 'aeiou'
    return {vowel: text_lower.count(vowel) for vowel in vowels}


# =============================================================================
# 6. File System and OS Interaction
# =============================================================================

def split_path(file_path: str) -> Tuple[str, str]:
    """
    Splits a file path into its directory and filename components.
    Uses os.path for robust, cross-platform behavior.

    Args:
        file_path: The full path to the file.
    
    Returns:
        A tuple containing (directory_path, filename).
    """
    if not isinstance(file_path, str):
        raise TypeError("Input path must be a string.")
    return os.path.split(file_path)

def join_path(base_dir: str, *args: str) -> str:
    """
    Joins directory and file names to create a valid path.
    Uses os.path.join for robust, cross-platform behavior.

    Args:
        base_dir: The starting directory.
        *args: Additional path components to join.

    Returns:
        A complete, joined file path.
    """
    if not all(isinstance(part, str) for part in (base_dir, *args)):
        raise TypeError("All path components must be strings.")
    return os.path.join(base_dir, *args)

def find_and_move_files(src_dir: str, dst_dir: str, pattern: str) -> None:
    """
    Finds all files matching a pattern in a source directory and moves them
    to a destination directory.

    Args:
        src_dir: The source directory path.
        dst_dir: The destination directory path.
        pattern: A glob pattern (e.g., "*.txt").

    Raises:
        FileNotFoundError: If source or destination directory does not exist.
    """
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source directory not found: {src_dir}")
    if not os.path.isdir(dst_dir):
        raise FileNotFoundError(f"Destination directory not found: {dst_dir}")

    matching_files = glob.glob(os.path.join(src_dir, pattern))
    
    if not matching_files:
        print(f"No files with pattern '{pattern}' found in '{src_dir}'.")
        return

    for file_path in matching_files:
        base_name = os.path.basename(file_path)
        os.rename(file_path, os.path.join(dst_dir, base_name))
    print(f"Moved {len(matching_files)} files.")


# =============================================================================
# 7. Main Demonstration Function
# =============================================================================

def main():
    """Main function to demonstrate the refactored code examples."""
    print("--- Refactored Python Code Examples ---")

    # --- Mathematical Functions ---
    print("\n--- Mathematical Functions ---")
    a, b = 54, 24
    print(f"GCD of {a} and {b} is: {gcd(a, b)}")
    print(f"LCM of {a} and {b} is: {lcm(a, b)}")

    num_for_factors = 36
    print(f"Factors of {num_for_factors} are: {get_factors(num_for_factors)}")

    # --- Simple Number Properties ---
    print("\n--- Simple Number Properties ---")
    print(f"Is 29 a prime number? {is_prime(29)}")
    print(f"Is 28 a prime number? {is_prime(28)}")
    
    print("Prime numbers between 900 and 1000:")
    primes_in_range = [num for num in range(900, 1001) if is_prime(num)]
    print(primes_in_range)
    
    year = 2000
    is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    print(f"Is {year} a leap year? {is_leap}")

    # --- Recursive Functions ---
    print("\n--- Recursive Functions ---")
    n_terms = 10
    print(f"Fibonacci sequence (first {n_terms} terms):")
    fib_seq = [fibonacci_recursive(i) for i in range(n_terms)]
    print(fib_seq)

    num_for_sum = 16
    print(f"Sum of natural numbers up to {num_for_sum}: {sum_natural_recursive(num_for_sum)}")

    num_for_fact = 7
    print(f"Factorial of {num_for_fact}: {factorial_recursive(num_for_fact)}")

    # --- Calculator ---
    # run_simple_calculator() # Uncomment to run the interactive calculator

    # --- Matrix Operations ---
    print("\n--- Matrix Operations ---")
    X = [[12, 7, 3], [4, 5, 6], [7, 8, 9]]
    Y = [[5, 8, 1], [6, 7, 3], [4, 5, 9]]
    print(f"Matrix X: {X}")
    print(f"Matrix Y: {Y}")
    print(f"Matrix Addition (X+Y): {add_matrices(X, Y)}")
    
    M = [[12, 7], [4, 5], [3, 8]]
    print(f"Matrix M: {M}")
    print(f"Transpose of M: {transpose_matrix(M)}")
    
    # --- String Manipulation ---
    print("\n--- String Manipulation ---")
    palindrome_str = 'aIbohPhoBiA'
    print(f"Is '{palindrome_str}' a palindrome? {is_palindrome(palindrome_str)}")
    
    punct_str = "Hello!!!, he said ---and went."
    print(f"Original string: '{punct_str}'")
    print(f"Without punctuation: '{remove_punctuation(punct_str)}'")

    sentence_to_sort = "Hello this Is an Example With cased letters"
    print(f"Original sentence: '{sentence_to_sort}'")
    print(f"Sorted words: {sort_words_alphabetically(sentence_to_sort)}")
    
    vowel_str = 'Hello, have you tried our tutorial section yet?'
    print(f"Vowel counts in '{vowel_str}': {count_vowels(vowel_str)}")

    # --- Set Operations ---
    print("\n--- Set Operations ---")
    E = {0, 2, 4, 6, 8}
    N = {1, 2, 3, 4, 5}
    print(f"Set E: {E}")
    print(f"Set N: {N}")
    print(f"Union (E | N): {E | N}")
    print(f"Intersection (E & N): {E & N}")
    print(f"Difference (E - N): {E - N}")
    print(f"Symmetric Difference (E ^ N): {E ^ N}")


if __name__ == "__main__":
    main()
