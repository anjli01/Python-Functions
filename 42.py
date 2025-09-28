# -*- coding: utf-8 -*-
"""
A collection of refactored Python code examples demonstrating standard practices.

This script covers various topics including:
- Generators and iterators
- List comprehensions
- Standard library modules (math, random, zlib, timeit, itertools)
- Basic algorithms (binary search)
- Object-Oriented Programming (OOP) concepts
- String and data structure manipulation
- Matrix operations
"""

import itertools
import math
import random
import timeit
import zlib
from collections import Counter
from typing import Generator, List, Any, Tuple, Union, Set, Dict


# ==============================================================================
# 1. GENERATOR EXAMPLES
# ==============================================================================

def generate_even_numbers(limit: int) -> Generator[int, None, None]:
    """
    Generates even numbers from 0 up to a given limit (inclusive).

    Args:
        limit: The maximum number to check.

    Yields:
        Even numbers between 0 and limit.
    """
    for i in range(0, limit + 1, 2):
        yield i


def generate_divisible_by_5_and_7(limit: int) -> Generator[int, None, None]:
    """
    Generates numbers divisible by both 5 and 7 from 0 to a limit.

    Args:
        limit: The maximum number to check.

    Yields:
        Numbers divisible by 35.
    """
    # A number divisible by 5 and 7 is divisible by their LCM, which is 35.
    for i in range(0, limit + 1, 35):
        yield i


# ==============================================================================
# 2. ALGORITHMS AND DATA STRUCTURES
# ==============================================================================

def binary_search(sorted_list: List[Any], item: Any) -> int:
    """
    Performs a binary search on a sorted list to find an item's index.

    Args:
        sorted_list: A list of items that is already sorted.
        item: The item to search for.

    Returns:
        The index of the item if found, otherwise -1.
    """
    low = 0
    high = len(sorted_list) - 1

    while low <= high:
        mid = (low + high) // 2  # Use integer division for simplicity
        guess = sorted_list[mid]
        if guess == item:
            return mid
        if guess > item:
            high = mid - 1
        else:
            low = mid + 1
    return -1


def remove_duplicates_preserve_order(items: List[Any]) -> List[Any]:
    """
    Removes duplicate items from a list while preserving original order.

    Args:
        items: A list that may contain duplicates.

    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def solve_chicken_and_rabbit_puzzle(
    total_heads: int, total_legs: int
) -> Union[Tuple[int, int], Tuple[str, str]]:
    """
    Solves the classic chicken and rabbit puzzle.

    Given a total number of heads and legs, finds the number of chickens (2 legs)
    and rabbits (4 legs).

    Args:
        total_heads: The total number of animals.
        total_legs: The total number of legs.

    Returns:
        A tuple of (chickens, rabbits) if a solution exists.
        Otherwise, a tuple indicating no solution was found.
    """
    for chickens in range(total_heads + 1):
        rabbits = total_heads - chickens
        if (2 * chickens) + (4 * rabbits) == total_legs:
            return chickens, rabbits
    return "No solution found", "No solution found"


# ==============================================================================
# 3. OBJECT-ORIENTED PROGRAMMING
# ==============================================================================

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


# ==============================================================================
# 4. STRING MANIPULATION
# ==============================================================================

def count_string_characters(input_string: str) -> Dict[str, int]:
    """
    Counts the occurrences of each character in a string.

    Args:
        input_string: The string to analyze.

    Returns:
        A dictionary mapping each character to its count.
    """
    # collections.Counter is the most Pythonic way to do this.
    return Counter(input_string)


def reverse_string(input_string: str) -> str:
    """
    Reverses a string using slicing.

    Args:
        input_string: The string to reverse.

    Returns:
        The reversed string.
    """
    return input_string[::-1]


def get_even_index_chars(input_string: str) -> str:
    """
    Extracts characters from a string that are at even indices.

    Args:
        input_string: The string to process.

    Returns:
        A new string containing characters from even indices.
    """
    return input_string[::2]


def count_vowels(input_string: str) -> int:
    """
    Counts the number of vowels in a given string (case-insensitive).

    Args:
        input_string: The string to check.

    Returns:
        The total count of vowels.
    """
    vowels = "aeiou"
    return sum(1 for char in input_string.lower() if char in vowels)


# ==============================================================================
# 5. MATRIX OPERATIONS
# ==============================================================================

Matrix = List[List[float]]

def print_matrix(matrix: Matrix, title: str = "Matrix"):
    """
    Prints a matrix in a readable, beautiful format.

    Args:
        matrix: The matrix (list of lists) to print.
        title: An optional title to print above the matrix.
    """
    print(f"\n--- {title} ---")
    if not matrix or not isinstance(matrix[0], list):
        print(matrix)  # Handle 1D lists or empty matrices
        return

    # Find the maximum width needed for any number in the matrix
    max_width = 0
    for row in matrix:
        for item in row:
            max_width = max(max_width, len(str(item)))
    
    for row in matrix:
        # Format each item with padding for alignment
        print("| " + " ".join(str(item).rjust(max_width) for item in row) + " |")


def create_zero_matrix(rows: int, cols: int) -> Matrix:
    """Creates a matrix of given dimensions filled with zeros."""
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def create_identity_matrix(size: int) -> Matrix:
    """Creates an identity matrix of a given size."""
    matrix = create_zero_matrix(size, size)
    for i in range(size):
        matrix[i][i] = 1.0
    return matrix


def transpose_matrix(matrix: Matrix) -> Matrix:
    """Transposes a given matrix."""
    if not matrix or not matrix[0]:
        return []
    rows, cols = len(matrix), len(matrix[0])
    return [[matrix[j][i] for j in range(rows)] for i in range(cols)]


def add_matrices(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    """Adds two matrices of the same dimensions."""
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])

    if (rows_a, cols_a) != (rows_b, cols_b):
        raise ValueError("Matrices must have the same dimensions for addition.")
    
    return [
        [matrix_a[i][j] + matrix_b[i][j] for j in range(cols_a)]
        for i in range(rows_a)
    ]

def subtract_matrices(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    """Subtracts matrix_b from matrix_a."""
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])

    if (rows_a, cols_a) != (rows_b, cols_b):
        raise ValueError("Matrices must have the same dimensions for subtraction.")
        
    return [
        [matrix_a[i][j] - matrix_b[i][j] for j in range(cols_a)]
        for i in range(rows_a)
    ]

def multiply_matrices(matrix_a: Matrix, matrix_b: Matrix) -> Matrix:
    """Multiplies two matrices."""
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])

    if cols_a != rows_b:
        raise ValueError("Number of columns in A must equal number of rows in B.")
    
    result = create_zero_matrix(rows_a, cols_b)
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a): # or range(rows_b)
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result


# ==============================================================================
# 6. MAIN DEMONSTRATION FUNCTION
# ==============================================================================

def main():
    """Main function to demonstrate all refactored code examples."""
    print("===== Python Code Examples Showcase =====")

    # --- Generator Examples ---
    print("\n--- Example: Even Number Generator ---")
    try:
        limit = int(input("Enter a number (n) to find evens up to: "))
        even_gen = generate_even_numbers(limit)
        print("Even numbers:", ",".join(map(str, even_gen)))
    except ValueError:
        print("Invalid input. Please enter an integer.")

    print("\n--- Example: Divisible by 5 and 7 Generator ---")
    try:
        limit = int(input("Enter a number (n) to find divisible by 5 & 7: "))
        num_gen = generate_divisible_by_5_and_7(limit)
        print("Numbers divisible by 5 and 7:", ",".join(map(str, num_gen)))
    except ValueError:
        print("Invalid input. Please enter an integer.")

    # --- Assertions ---
    print("\n--- Example: Asserting all numbers are even ---")
    even_list = [2, 4, 6, 8]
    for num in even_list:
        assert num % 2 == 0, f"{num} is not an even number!"
    print(f"Assertion passed: All numbers in {even_list} are even.")

    # --- Eval (with safety warning) ---
    print("\n--- Example: Evaluating a mathematical expression ---")
    print("WARNING: Using eval() with untrusted input is a security risk!")
    expression = input("Enter a simple math expression (e.g., 3 * (4 + 5)): ")
    try:
        result = eval(expression, {"__builtins__": {}}, {}) # Safely constrained eval
        print(f"The result of '{expression}' is: {result}")
    except Exception as e:
        print(f"Could not evaluate expression. Error: {e}")

    # --- Binary Search ---
    print("\n--- Example: Binary Search ---")
    sorted_numbers = [2, 5, 7, 9, 11, 17, 222]
    print(f"Searching in list: {sorted_numbers}")
    print(f"Index of 11: {binary_search(sorted_numbers, 11)}")  # Expected: 4
    print(f"Index of 12: {binary_search(sorted_numbers, 12)}")  # Expected: -1

    # --- Random Module Examples ---
    print("\n--- Example: Random Number Generation ---")
    print(f"Random float between 10 and 100: {random.uniform(10, 100)}")
    print(f"Random float between 5 and 95: {random.uniform(5, 95)}")
    print(f"Random even number (0-10): {random.choice(range(0, 11, 2))}")
    divisible_by_35 = [i for i in range(201) if i % 35 == 0]
    print(f"Random num divisible by 5&7 (0-200): {random.choice(divisible_by_35)}")
    print(f"5 random numbers (100-200): {random.sample(range(100, 201), 5)}")
    print(f"Random integer (7-15): {random.randint(7, 15)}")

    # --- Compression ---
    print("\n--- Example: Zlib Compression ---")
    original_string = b'hello world!' * 4
    print(f"Original: {original_string}")
    compressed = zlib.compress(original_string)
    print(f"Compressed: {compressed}")
    decompressed = zlib.decompress(compressed)
    print(f"Decompressed: {decompressed}")

    # --- Performance Timing ---
    print("\n--- Example: Timing Code Execution with timeit ---")
    execution_time = timeit.timeit("1 + 1", number=100000)
    print(f"Time to execute '1+1' 100,000 times: {execution_time:.6f} seconds")

    # --- List Shuffling ---
    print("\n--- Example: Shuffling a list ---")
    my_list = [3, 6, 7, 8]
    print(f"Original list: {my_list}")
    random.shuffle(my_list)
    print(f"Shuffled list: {my_list}")

    # --- Sentence Generation ---
    print("\n--- Example: Generating Sentences ---")
    subjects = ["I", "You"]
    verbs = ["Play", "Love"]
    objects = ["Hockey", "Football"]
    for sub, verb, obj in itertools.product(subjects, verbs, objects):
        print(f"{sub} {verb} {obj}.")

    # --- List Comprehensions for Filtering ---
    print("\n--- Example: List Filtering with Comprehensions ---")
    original_list = [12, 24, 35, 70, 88, 120, 155]
    print(f"Original list: {original_list}")
    print(f"Remove evens: {[x for x in [5, 6, 77, 45, 22, 12, 24] if x % 2 != 0]}")
    print(f"Remove divisible by 5&7: {[x for x in original_list if x % 35 != 0]}")
    print(f"Remove by index (0,2,4,6): {[x for i, x in enumerate(original_list) if i % 2 != 0]}")
    print(f"Remove by value (24): {[x for x in [12, 24, 35, 24, 88] if x != 24]}")

    # --- Set Operations ---
    print("\n--- Example: Set Intersection ---")
    list1 = [1, 3, 6, 78, 35, 55]
    list2 = [12, 24, 35, 24, 88, 120, 155]
    set1, set2 = set(list1), set(list2)
    print(f"Intersection of {list1} and {list2}: {list(set1.intersection(set2))}")

    # --- Removing Duplicates ---
    print("\n--- Example: Removing Duplicates (Order Preserved) ---")
    dup_list = [12, 24, 35, 24, 88, 120, 155, 88, 120, 155]
    print(f"Original list: {dup_list}")
    print(f"Deduplicated: {remove_duplicates_preserve_order(dup_list)}")

    # --- OOP Example ---
    print("\n--- Example: OOP with Inheritance ---")
    male = Male()
    female = Female()
    print(f"A 'Male' object's gender is: {male.get_gender()}")
    print(f"A 'Female' object's gender is: {female.get_gender()}")
    
    # --- String examples ---
    print("\n--- Example: Character Counting ---")
    test_string = "AmmarAdil"
    print(f"Character counts for '{test_string}': {count_string_characters(test_string)}")
    print("\n--- Example: Vowel Counting ---")
    print(f"Vowel count for '{test_string}': {count_vowels(test_string)}")

    # --- Matrix Operations Showcase ---
    print("\n" + "="*20 + " MATRIX OPERATIONS " + "="*20)
    matrix_a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    matrix_b = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    
    print_matrix(matrix_a, title="Matrix A")
    print_matrix(matrix_b, title="Matrix B")
    
    print_matrix(create_identity_matrix(3), title="3x3 Identity Matrix")
    print_matrix(transpose_matrix(matrix_a), title="Transpose of A")
    print_matrix(add_matrices(matrix_a, matrix_b), title="A + B")
    print_matrix(subtract_matrices(matrix_a, matrix_b), title="A - B")
    
    m1 = [[1, 2], [3, 4]]
    m2 = [[5, 6], [7, 8]]
    print_matrix(multiply_matrices(m1, m2), title="[[1,2],[3,4]] * [[5,6],[7,8]]")


if __name__ == "__main__":
    main()
