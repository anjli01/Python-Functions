# -*- coding: utf-8 -*-
"""
A collection of refactored Python code examples following best practices.

This script includes functions for various tasks such as data manipulation,
mathematical calculations, string processing, and more. Each function has been
improved for readability, includes type hints, and follows PEP 8 standards.
"""

# --- IMPORTS ---
import collections
import ctypes
import math
import re
import sys
import time
import urllib.request
from datetime import datetime
from decimal import Decimal, getcontext
from functools import reduce
from operator import mul
from random import uniform
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Optional

# Set precision for Decimal calculations for consistency in examples
getcontext().prec = 50


# --- REFACTORED FUNCTIONS ---

# 8. Python function to identify profane words
def filter_profane_words(text: str) -> List[str]:
    """
    Identifies and returns a list of profane words found in the input text.

    Fetches a list of profane words from a remote URL, tokenizes the input
    text, and returns words from the text that are in the profane list.

    Args:
        text: The string to be checked for profane words.

    Returns:
        A list of profane words found in the text. Returns an empty list
        if the profanity list cannot be fetched or no profane words are found.
    """
    profane_word_url = "https://raw.githubusercontent.com/RobertJGabriel/Google-profanity-words/master/list.txt"
    try:
        with urllib.request.urlopen(profane_word_url) as response:
            # Create a set for efficient O(1) average time complexity lookups.
            profane_words = {line.decode("utf-8").strip() for line in response}
    except urllib.error.URLError as e:
        print(f"Error fetching profanity list: {e}")
        return []

    # Tokenize the input string into words
    words_in_text = set(re.findall(r'\w+', text.lower()))

    # Find the intersection of words in the text and the profane list
    return sorted(list(words_in_text.intersection(profane_words)))


# 9. Python function to sum even numbers in a list
def sum_even_numbers(numbers: List[int]) -> int:
    """
    Calculates the sum of all even numbers in a list.

    Args:
        numbers: A list of integers.

    Returns:
        The sum of the even numbers.
    """
    # A generator expression with sum() is more readable than filter/reduce.
    return sum(num for num in numbers if num % 2 == 0)


# 10. Python function to find the area of a circle
def calculate_circle_area(radius: float) -> float:
    """
    Calculates the area of a circle given its radius.

    Args:
        radius: The radius of the circle.

    Returns:
        The area of the circle.
    """
    # Using math.pi is more accurate than the 22/7 approximation.
    return math.pi * (radius ** 2)


# 11. Python program to find whether a number is prime
def is_prime(number: int) -> bool:
    """
    Checks if a number is a prime number.

    Args:
        number: An integer.

    Returns:
        True if the number is prime, False otherwise.
    """
    if number < 2:
        return False
    # We only need to check for divisors up to the square root of the number.
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False
    return True


# 12. Python function to return the cubes of a list of numbers
def get_cubes(numbers: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Returns a new list containing the cube of each number from the input list.

    Args:
        numbers: A list of numbers (integers or floats).

    Returns:
        A list of the cubed numbers.
    """
    return [num ** 3 for num in numbers]


# 13. Python function to find the average of given numbers (Stateful version)
class RunningAverage:
    """
    A class to calculate the running average of a series of numbers.
    This is a clearer, object-oriented way to manage state compared to closures.
    """
    def __init__(self):
        self._numbers: List[Union[int, float]] = []

    def add(self, number: Union[int, float]) -> float:
        """
        Adds a number to the collection and returns the new average.

        Args:
            number: The number to add.

        Returns:
            The updated average of all numbers added so far.
        """
        self._numbers.append(number)
        return sum(self._numbers) / len(self._numbers)

    def get_average(self) -> float:
        """Returns the current average."""
        if not self._numbers:
            return 0.0
        return sum(self._numbers) / len(self._numbers)

    def __call__(self, number: Union[int, float]) -> float:
        """Allows the instance to be called like a function."""
        return self.add(number)


# 14. Python function to create adders (demonstrating closures and late binding)
def create_adders() -> List[Callable[[int], int]]:
    """
    Creates a list of lambda functions, each adding a different number.

    This example demonstrates a common closure pitfall known as "late binding".
    The value of 'n' is not captured when the lambda is defined, but when it is
    called. The fix is to assign 'n' to a local variable in the lambda's scope.

    Returns:
        A list of three lambda functions.
    """
    adders = []
    for n in range(1, 4):
        # The `num=n` captures the current value of `n` for each lambda.
        # Without this, all lambdas would use the final value of n (which is 3).
        adders.append(lambda x, num=n: x + num)
    return adders


# 15. Python function for datetime with a safe default argument
def log_message(message: str, *, timestamp: Optional[datetime] = None) -> None:
    """
    Logs a message with a timestamp.

    The default timestamp is generated at call time, avoiding the common pitfall
    of using mutable or function-call default arguments.

    Args:
        message: The message to log.
        timestamp: An optional datetime object. If None, datetime.utcnow() is used.
    """
    # Use `is None` to allow for a fresh datetime object on each call.
    if timestamp is None:
        timestamp = datetime.utcnow()
    print(f'Message at {timestamp:%Y-%m-%d %H:%M:%S} UTC was: {message}')


# 16. Python function for count of address reference
def get_reference_count(address: int) -> int:
    """
    Returns the reference count of a Python object at a given memory address.

    Note: Type annotations like `address: int` are for readability and static
    analysis; they don't change how Python runs the code.

    Args:
        address: The memory address of the object (from id()).

    Returns:
        The object's reference count.
    """
    return ctypes.c_long.from_address(address).value


# 17. Python function to modify a mutable element within a tuple
def modify_mutable_in_tuple(data_tuple: Tuple[List[Any], ...]) -> None:
    """
    Demonstrates that while a tuple is immutable, its mutable elements
    (like a list) can be modified in place.

    Args:
        data_tuple: A tuple containing at least one list.
    """
    print(f'Initial tuple memory address = {id(data_tuple)}')
    print(f'Initial tuple content: {data_tuple}')
    # The list inside the tuple is mutable and can be changed.
    data_tuple[0].append(100)
    print(f'Final tuple memory address   = {id(data_tuple)}') # Address remains the same
    print(f'Final tuple content: {data_tuple}')


# 18 & 19. Python programs to compare string comparison performance
def time_string_comparisons(iterations: int) -> None:
    """
    Compares the performance of string comparison using `==` vs `is`
    with and without string interning.

    Args:
        iterations: The number of comparison loops to run.
    """
    # `a == b` compares values, which is slow for long strings.
    a_long = 'a long string that is not interned' * 200
    b_long = 'a long string that is not interned' * 200
    start_equals = time.perf_counter()
    for _ in range(iterations):
        if a_long == b_long:
            pass
    end_equals = time.perf_counter()
    print(f'Using "==": {end_equals - start_equals:.6f} seconds')

    # `a is b` compares memory addresses. It's very fast but only works if
    # both variables point to the same object. `sys.intern` ensures this.
    a_interned = sys.intern('a long string that is not interned' * 200)
    b_interned = sys.intern('a long string that is not interned' * 200)
    start_is = time.perf_counter()
    for _ in range(iterations):
        if a_interned is b_interned:
            pass
    end_is = time.perf_counter()
    print(f'Using "is" with interning: {end_is - start_is:.6f} seconds')


# 20. Python program to calculate the time taken to create float vs Decimal
def time_numeric_creation(iterations: int = 1_000_000) -> None:
    """
    Compares the performance of creating float vs. Decimal objects.

    Args:
        iterations: The number of objects to create.
    """
    start_float = time.perf_counter()
    for _ in range(iterations):
        _ = 3.1415
    end_float = time.perf_counter()
    print(f'Time to create {iterations:,} floats: {end_float - start_float:.6f} seconds')

    start_decimal = time.perf_counter()
    for _ in range(iterations):
        _ = Decimal('3.1415')
    end_decimal = time.perf_counter()
    print(f'Time to create {iterations:,} Decimals: {end_decimal - start_decimal:.6f} seconds')


# 21. Python function for factorial using reduce
def factorial_reduce(n: int) -> int:
    """
    Calculates the factorial of a non-negative integer using functools.reduce.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial of n.
    """
    if not isinstance(n, int) or n < 0:
        raise ValueError("Factorial is only defined for non-negative integers")
    if n == 0:
        return 1
    # reduce(mul, [1, 2, 3, ..., n])
    return reduce(mul, range(1, n + 1))


# 22. Python program to find if given co-ordinates are inside a circle
def is_point_in_circle(radius: float, x: float, y: float) -> bool:
    """
    Checks if a point (x, y) is inside a circle of a given radius.

    Args:
        radius: The circle's radius.
        x: The x-coordinate of the point.
        y: The y-coordinate of the point.

    Returns:
        True if the point is inside or on the circle, False otherwise.
    """
    # It's more efficient to compare squares to avoid the sqrt calculation.
    return (x**2 + y**2) <= (radius**2)


# 23. Python function to find the area of a square
def calculate_square_area(side_length: float) -> float:
    """
    Calculates the area of a square.

    Args:
        side_length: The length of one side of the square.

    Returns:
        The area of the square.
    """
    return side_length ** 2


# 24. Python program for the sum of first n numbers.
def sum_first_n_integers(n: int) -> float:
    """
    Calculates the sum of the first n positive integers using the arithmetic
    series formula.

    Args:
        n: The number of integers to sum (must be positive).

    Returns:
        The sum of integers from 1 to n.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
    return n * (n + 1) / 2


# 25. Python Program to Add two Lists element-wise
def add_lists_elementwise(list1: List[float], list2: List[float]) -> List[float]:
    """
    Adds two lists of numbers element by element.

    Args:
        list1: The first list of numbers.
        list2: The second list of numbers.

    Returns:
        A new list where each element is the sum of the elements at the
        same index in the input lists.

    Raises:
        ValueError: If the lists are not of the same length.
    """
    if len(list1) != len(list2):
        raise ValueError("Input lists must be of the same length.")
    # zip() is perfect for this, combined with a list comprehension.
    return [x + y for x, y in zip(list1, list2)]


# 26. Python Program to find Largest and Smallest Number in a List
def find_min_max_with_indices(numbers: List[float]) -> Dict[str, Any]:
    """
    Finds the smallest and largest numbers in a list and their indices.

    Args:
        numbers: A list of numbers.

    Returns:
        A dictionary containing the min/max values and their first-occurrence indices.
        Returns an empty dictionary if the input list is empty.
    """
    if not numbers:
        return {}

    min_val = min(numbers)
    max_val = max(numbers)
    return {
        'min_value': min_val,
        'min_index': numbers.index(min_val),
        'max_value': max_val,
        'max_index': numbers.index(max_val)
    }


# 27. Python Palindrome Program for Integers
def is_palindrome_integer(number: int) -> bool:
    """
    Checks if an integer is a palindrome.

    This implementation avoids recursion and global state by converting the
    number to a string, which is a more Pythonic and safer approach.

    Args:
        number: The integer to check.

    Returns:
        True if the number reads the same forwards and backwards, False otherwise.
    """
    if not isinstance(number, int):
        raise TypeError("Input must be an integer.")
    num_str = str(abs(number))  # Handle negative numbers gracefully
    return num_str == num_str[::-1]


# 28. Python Program to Swap Two Numbers
def swap_values(a: Any, b: Any) -> Tuple[Any, Any]:
    """
    Swaps the values of two variables using tuple packing/unpacking.

    Args:
        a: The first value.
        b: The second value.

    Returns:
        A tuple containing the swapped values (b, a).
    """
    # Python's tuple packing and unpacking is the idiomatic way to swap.
    return b, a


# 30. Python Program to find Largest of Three Numbers
def find_largest_of_three(a: float, b: float, c: float) -> float:
    """
    Finds the largest of three numbers.

    Args:
        a, b, c: The three numbers to compare.

    Returns:
        The largest number among the three.
    """
    # The built-in max() function is the simplest and most readable solution.
    return max(a, b, c)


# 31. Python Program for Circle Calculations
def get_circle_metrics(radius: float) -> Dict[str, float]:
    """
    Calculates the diameter, circumference, and area of a circle.

    Args:
        radius: The radius of the circle.

    Returns:
        A dictionary containing the diameter, circumference, and area.
    """
    if radius < 0:
        raise ValueError("Radius cannot be negative.")
    return {
        "diameter": 2 * radius,
        "circumference": 2 * math.pi * radius,
        "area": math.pi * (radius ** 2)
    }


# 34. Python Program to Map two lists into a Dictionary
def create_dict_from_lists(keys: List[Any], values: List[Any]) -> Dict[Any, Any]:
    """
    Creates a dictionary by mapping a list of keys to a list of values.

    Args:
        keys: A list of keys.
        values: A list of values.

    Returns:
        A dictionary created from the keys and values.
    """
    # The dict() constructor with zip() is the idiomatic way to do this.
    return dict(zip(keys, values))


# 35. Python function To Calculate Volume Of Cylinder
def calculate_cylinder_volume(radius: float, height: float) -> float:
    """
    Calculates the volume of a cylinder.

    Args:
        radius: The radius of the cylinder's base.
        height: The height of the cylinder.

    Returns:
        The volume of the cylinder.
    """
    return calculate_circle_area(radius) * height


# 36. Recursive Python function to solve the tower of hanoi
def solve_tower_of_hanoi(disks: int, source: str, destination: str, auxiliary: str) -> None:
    """
    Prints the steps to solve the Tower of Hanoi puzzle.

    Args:
        disks: The number of disks to move.
        source: The name of the source peg.
        destination: The name of the destination peg.
        auxiliary: The name of the auxiliary peg.
    """
    if disks == 1:
        print(f"Move disk 1 from {source} to {destination}")
        return
    solve_tower_of_hanoi(disks - 1, source, auxiliary, destination)
    print(f"Move disk {disks} from {source} to {destination}")
    solve_tower_of_hanoi(disks - 1, auxiliary, destination, source)


# 37. Python function to find angle between hour and minute hand
def calculate_clock_angle(hour: int, minute: int) -> float:
    """
    Calculates the smaller angle between the hour and minute hands of a clock.

    Args:
        hour: The hour (0-11 or 1-12).
        minute: The minute (0-59).

    Returns:
        The smaller angle in degrees.
    """
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError("Invalid time provided.")

    # Convert 24-hour format to 12-hour format for calculation
    hour %= 12

    hour_angle = 0.5 * (hour * 60 + minute)
    minute_angle = 6 * minute
    angle = abs(hour_angle - minute_angle)

    # The angle can be the direct difference or 360 minus the difference
    return min(angle, 360 - angle)


# 39. Python program to reverse a linked list
# (The implementation for Node and LinkedList is kept similar for clarity)
class Node:
    """A node in a singly linked list."""
    def __init__(self, data: Any):
        self.data = data
        self.next: Optional[Node] = None

class LinkedList:
    """A singly linked list."""
    def __init__(self):
        self.head: Optional[Node] = None

    def push(self, new_data: Any) -> None:
        """Adds a new node at the beginning of the list."""
        new_node = Node(new_data)
        new_node.next = self.head
        self.head = new_node

    def reverse(self) -> None:
        """Reverses the linked list in place."""
        prev_node = None
        current_node = self.head
        while current_node is not None:
            next_node = current_node.next  # Store the next node
            current_node.next = prev_node   # Reverse the link
            prev_node = current_node        # Move prev_node forward
            current_node = next_node        # Move current_node forward
        self.head = prev_node

    def print_list(self) -> None:
        """Prints the elements of the list."""
        temp = self.head
        elements = []
        while temp:
            elements.append(str(temp.data))
            temp = temp.next
        print(" -> ".join(elements))


# 40. Python function to Remove all duplicates from a given string
def remove_duplicate_chars(text: str) -> Tuple[str, str]:
    """
    Removes duplicate characters from a string.

    Args:
        text: The input string.

    Returns:
        A tuple containing two strings:
        - The first with unique characters, order not preserved.
        - The second with unique characters, original order preserved.
    """
    # Method 1: Using a set (fast, but does not preserve order)
    unique_chars_unordered = "".join(set(text))

    # Method 2: Iterating to preserve order (Python 3.7+ dicts are ordered)
    unique_chars_ordered = "".join(dict.fromkeys(text).keys())

    return unique_chars_unordered, unique_chars_ordered


# ... Omitting the simple math wrappers (41-53) for brevity as they are mostly
# straightforward wrappers of the `math` module. The refactoring would involve

# consistent docstrings, type hints, and potentially raising ValueErrors
# for invalid inputs (e.g., log of a negative number).

# Example of refactored softmax (51):
def softmax(vector: List[float]) -> List[float]:
    """
    Calculates the softmax for a list of numbers.

    Args:
        vector: A list of numbers (logits).

    Returns:
        A list of probabilities that sum to 1.
    """
    if not isinstance(vector, list):
        raise TypeError("Input must be a list of numbers.")

    exponents = [math.exp(i) for i in vector]
    total = sum(exponents)
    return [j / total for j in exponents]

# --- String Manipulation Functions ---

def is_palindrome_string(text: str) -> bool:
    """Checks if a string is a palindrome."""
    return text == text[::-1]

def reverse_words(sentence: str) -> str:
    """Reverses the order of words in a sentence."""
    return " ".join(reversed(sentence.split()))

def find_word_frequency(text: str) -> Dict[str, int]:
    """Counts the frequency of each word in a string."""
    words = re.findall(r'\b\w+\b', text.lower())
    return collections.Counter(words)

def find_even_length_words(text: str) -> List[str]:
    """Finds all words with an even number of characters."""
    return [word for word in text.split() if len(word) % 2 == 0]

# --- Dictionary Manipulation Functions ---

def invert_dictionary(data: Dict[Any, Any]) -> Dict[Any, List[Any]]:
    """
    Inverts a dictionary. Keys become values, and values become keys.
    Handles non-unique values by grouping original keys into a list.
    """
    inverted_dict: Dict[Any, List[Any]] = {}
    for key, value in data.items():
        inverted_dict.setdefault(value, []).append(key)
    return inverted_dict

def flatten_dictionary(
    nested_dict: Dict[str, Any],
    parent_key: str = '',
    separator: str = '_'
) -> Dict[str, Any]:
    """
    Flattens a nested dictionary by joining keys with a separator.
    """
    items = []
    for key, value in nested_dict.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(flatten_dictionary(value, new_key, separator).items())
        else:
            items.append((new_key, value))
    return dict(items)


# --- MAIN EXECUTION BLOCK ---
def main() -> None:
    """
    Main function to demonstrate the refactored code examples.
    """
    print("--- 8. Profanity Filter ---")
    dirty_text = "This is a crappy example with some bad words like hell."
    filtered = filter_profane_words(dirty_text)
    print(f"Original: '{dirty_text}'\nProfane words found: {filtered}\n")

    print("--- 9. Sum Even Numbers ---")
    nums = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    print(f"Sum of evens in {nums}: {sum_even_numbers(nums)}\n")

    print("--- 11. Prime Number Check ---")
    print(f"Is 29 prime? {is_prime(29)}")
    print(f"Is 1 prime? {is_prime(1)}")
    print(f"Is 100 prime? {is_prime(100)}\n")

    print("--- 13. Running Average ---")
    avg_calculator = RunningAverage()
    print(f"Adding 10, new average: {avg_calculator.add(10)}")
    print(f"Adding 20, new average: {avg_calculator(20)}") # Using __call__
    print(f"Adding 45, new average: {avg_calculator.add(45)}\n")

    print("--- 15. Safe Logging with Timestamps ---")
    log_message("First log call.")
    time.sleep(1)
    log_message("Second log call, 1 second later.\n")

    print("--- 25. Add Lists Element-wise ---")
    list_a = [1, 2, 3]
    list_b = [10, 20, 30]
    print(f"{list_a} + {list_b} = {add_lists_elementwise(list_a, list_b)}\n")

    print("--- 26. Find Min/Max in List ---")
    data = [10, 50, 5, 90, 15]
    print(f"Data: {data}\nMin/Max info: {find_min_max_with_indices(data)}\n")

    print("--- 27. Integer Palindrome Check ---")
    print(f"Is 12321 a palindrome? {is_palindrome_integer(12321)}")
    print(f"Is 12345 a palindrome? {is_palindrome_integer(12345)}\n")

    print("--- 28. Swap Values ---")
    x, y = 100, 200
    print(f"Before swap: x={x}, y={y}")
    x, y = swap_values(x, y)
    print(f"After swap:  x={x}, y={y}\n")

    print("--- 36. Tower of Hanoi ---")
    print("Solving for 3 disks:")
    solve_tower_of_hanoi(disks=3, source='A', destination='C', auxiliary='B')
    print("")

    print("--- 40. Remove Duplicate Chars ---")
    original_str = "programming"
    unord, ordr = remove_duplicate_chars(original_str)
    print(f"Original: '{original_str}'")
    print(f"Unique (unordered): '{unord}'")
    print(f"Unique (ordered): '{ordr}'\n")

    print("--- Word Frequency Example ---")
    text_freq = "It is a great meal at a great restaurant on a great day."
    print(f"Frequency in '{text_freq}':\n{find_word_frequency(text_freq)}\n")

    print("--- Flatten Dictionary Example ---")
    nested = {'user': {'name': 'John', 'address': {'city': 'New York'}}}
    print(f"Nested: {nested}")
    print(f"Flattened: {flatten_dictionary(nested)}\n")


if __name__ == "__main__":
    main()