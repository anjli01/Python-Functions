# -*- coding: utf-8 -*-
"""
A collection of refactored Python code examples demonstrating various concepts,
from basic syntax to algorithms and data structures. Each example has been
improved for readability, adherence to standard practices (PEP 8), and clarity.
"""

# =============================================================================
# 1. Imports
# =============================================================================
# Standard library imports are grouped at the top of the file.
import cmath
import collections
import csv
import datetime
import dis
import hashlib
import itertools
import math
import os
import random
import re
import string
import time
from collections import Counter, deque
from functools import reduce
from html.parser import HTMLParser
from typing import (Any, Callable, Dict, Iterable, List, Optional, Set, Tuple,
                    Type, Union)

# Third-party imports would go here, with a blank line separating them.
# For example:
# import nltk
# from cryptography.fernet import Fernet


# =============================================================================
# 2. Core Python Concepts
# =============================================================================

def demonstrate_random_choice() -> None:
    """Shows how to randomly select and call a type constructor."""
    print("--- Random Choice Demonstration ---")
    type_options: List[Union[Type[float], Type[int], Type[str]]] = [float, int, str]
    for i in range(5):
        chosen_type: Union[Type[float], Type[int], Type[str]] = random.choice(type_options)
        # Dynamically call the chosen type's constructor (e.g., float(i))
        value = chosen_type(i)
        print(f"Value: {value!r}, Type: {type(value)}")


def demonstrate_generator_lazy_evaluation() -> None:
    """
    Demonstrates the lazy evaluation of generator expressions.
    The expression inside a generator is only evaluated when an item is requested.
    """
    print("\n--- Generator Lazy Evaluation ---")
    # This will fail immediately because the iterable `1/0` is evaluated at
    # definition time.
    print("Attempting to define a generator with an immediate error...")
    try:
        gen_fails = (i for i in 1 / 0)
    except ZeroDivisionError:
        print("Caught expected ZeroDivisionError on generator definition.")

    # This definition succeeds because `1/0` is only evaluated during iteration,
    # which we control.
    gen_succeeds = (i for i in range(5) for j in 1 / 0)
    print("\nGenerator defined successfully. It will fail upon iteration...")
    try:
        for _ in gen_succeeds:
            pass  # This line will never be reached
    except ZeroDivisionError:
        print("Caught expected ZeroDivisionError during iteration.")


def demonstrate_args(*args: Any) -> None:
    """
    Demonstrates the usage of *args to accept a variable number of positional
    arguments, which are packed into a tuple.
    """
    print("\n--- *args Demonstration ---")
    print(f"Type of args: {type(args)}")
    print(f"Args contents: {args}")
    if args:
        print(f"First argument: {args[0]}")


def demonstrate_kwargs(**kwargs: Any) -> None:
    """
    Demonstrates the usage of **kwargs to accept a variable number of keyword
    arguments, which are packed into a dictionary.
    """
    print("\n--- **kwargs Demonstration ---")
    print(f"Type of kwargs: {type(kwargs)}")
    print(f"kwargs contents: {kwargs}")
    if 'a' in kwargs:
        print(f"Value of argument 'a': {kwargs['a']}")


def demonstrate_iterable_unpacking() -> None:
    """Shows extended iterable unpacking using the * operator."""
    print("\n--- Iterable Unpacking ---")
    numbers = [1, 2, 3, 4, 5]
    first, *middle, last = numbers
    print(f"Original list: {numbers}")
    print(f"First element: {first}")
    print(f"Middle elements: {middle}")
    print(f"Last element: {last}")


def demonstrate_for_else_loop() -> None:
    """
    Demonstrates the `for...else` construct. The `else` block executes only
    if the loop completes without encountering a `break` statement.
    """
    print("\n--- For...Else Loop Demonstration ---")
    print("Loop that will complete (no break):")
    for i in range(3):
        print(f"  - Iteration {i}")
    else:
        print("  - Loop completed, so the 'else' block is executed.")

    print("\nLoop that will be interrupted (with break):")
    for i in range(3):
        print(f"  - Iteration {i}")
        if i == 1:
            print("  - 'break' statement encountered.")
            break
    else:
        # This block will not be executed.
        print("  - This 'else' block will NOT be executed.")


def demonstrate_while_else_loop() -> None:
    """
    Demonstrates the `while...else` construct. The `else` block executes
    only if the loop terminates because its condition became false.
    """
    print("\n--- While...Else Loop Demonstration ---")
    i = 0
    while i < 3:
        print(f"  - i = {i}")
        i += 1
    else:
        print("  - Loop condition became false, so 'else' block executed.")


def demonstrate_string_interning() -> None:
    """
    Demonstrates string interning, a CPython optimization where some strings
    are stored only once in memory.
    """
    print("\n--- String Interning ---")
    # 'hello_1' and 'hello_2' are compile-time constants, so Python interns them,
    # and they point to the same memory object.
    hello_1 = 'Hello'
    hello_2 = 'Hell' + 'o'
    print(f"'hello_1' is 'hello_2': {hello_1 is hello_2} (interned at compile time)")

    # 'hello_3' is created at runtime, which CPython may or may not intern.
    # Typically, such concatenations are not interned.
    base = 'Hell'
    hello_3 = base + 'o'
    print(f"'hello_1' is 'hello_3': {hello_1 is hello_3} (created at runtime)")


def demonstrate_disassembler() -> None:
    """Shows how to use the `dis` module to inspect Python bytecode."""
    print("\n--- Disassembler Demonstration ---")

    def simple_func():
        s = 'Hello'
        return s

    print("Bytecode for simple_func:")
    dis.dis(simple_func)


# =============================================================================
# 3. Functions and Algorithms
# =============================================================================

def greet(name: str) -> None:
    """
    Greets the person with the given name.

    Args:
        name: The name of the person to greet.
    """
    print(f"Hello, {name}. Good morning!")


def absolute_value(num: Union[int, float]) -> Union[int, float]:
    """
    Returns the absolute value of a number.

    Args:
        num: The number to find the absolute value of.

    Returns:
        The absolute value.
    """
    return num if num >= 0 else -num


def catalan(n: int) -> int:
    """
    Calculates the nth Catalan number using a naive recursive approach.
    Note: This is computationally expensive. A dynamic programming or
    memoization approach is better for larger n.

    Args:
        n: The index of the Catalan number to find (must be non-negative).

    Returns:
        The nth Catalan number.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    if n <= 1:
        return 1

    result = 0
    for i in range(n):
        result += catalan(i) * catalan(n - i - 1)
    return result


def binomial_coefficient(n: int, k: int) -> int:
    """
    Calculates the binomial coefficient C(n, k) using recursion.
    Note: Inefficient for large inputs. `math.comb(n, k)` (Python 3.8+)
    or a dynamic programming approach is preferred.

    Args:
        n: The total number of items.
        k: The number of items to choose.

    Returns:
        The binomial coefficient "n choose k".
    """
    if k > n or k < 0:
        return 0
    if k == 0 or k == n:
        return 1
    # C(n, k) = C(n-1, k-1) + C(n-1, k)
    return binomial_coefficient(n - 1, k - 1) + binomial_coefficient(n - 1, k)


def find_longest_increasing_subsequence(arr: List[int]) -> int:
    """
    Finds the length of the Longest Increasing Subsequence (LIS) using a
    naive recursive approach without global variables.

    This implementation is for demonstration and has exponential time complexity.
    A dynamic programming solution with O(n^2) or O(n log n) is standard.

    Args:
        arr: A list of numbers.

    Returns:
        The length of the LIS.
    """
    n = len(arr)
    if n == 0:
        return 0

    memo = {}

    def _lis_ending_at(index: int) -> int:
        """Helper to find LIS length ending at a specific index."""
        if index in memo:
            return memo[index]

        max_len = 1
        for i in range(index):
            # If a previous element is smaller, it can be part of a sequence
            # ending at the current element.
            if arr[i] < arr[index]:
                max_len = max(max_len, 1 + _lis_ending_at(i))
        
        memo[index] = max_len
        return max_len

    # The overall LIS is the maximum of LIS ending at any position.
    return max( _lis_ending_at(i) for i in range(n) )


def fibonacci_recursive(n: int) -> int:
    """
    Calculates the nth Fibonacci number using recursion.
    Note: Highly inefficient due to repeated calculations.
    An iterative approach is much better.

    Args:
        n: The index of the Fibonacci number (non-negative).

    Returns:
        The nth Fibonacci number.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def concatenate_two_lists(list1: List[Any], list2: List[Any]) -> List[Any]:
    """
    Concatenates two lists.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        A new list containing elements from both lists.
    """
    return list1 + list2


def add_from_lists_conditionally(
    list1: List[int], list2: List[int]
) -> List[int]:
    """
    Creates a new list by summing elements from two lists where the element
    from the first list is even and the element from the second is odd.

    Args:
        list1: The first list of integers.
        list2: The second list of integers.

    Returns:
        A new list with the conditional sums.
    """
    return [
        x + y for x, y in zip(list1, list2) if x % 2 == 0 and y % 2 != 0
    ]


def kmh_to_mph(kmh: float) -> float:
    """Converts speed from kilometers per hour (KM/H) to miles per hour (MPH)."""
    KMH_TO_MPH_FACTOR = 0.621371
    return kmh * KMH_TO_MPH_FACTOR


def custom_sort(data: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Sorts a list of numbers using the selection sort algorithm.
    Note: This is for demonstration. For practical use, Python's built-in
    `sorted()` function or `.sort()` method are highly optimized and preferred.

    Args:
        data: A list of numbers to sort.

    Returns:
        A new list containing the sorted numbers.
    """
    source_list = data.copy()  # Avoid modifying the original list
    sorted_list = []
    while source_list:
        minimum = source_list[0]
        for item in source_list:
            if item < minimum:
                minimum = item
        sorted_list.append(minimum)
        source_list.remove(minimum)  # Note: remove() is an O(n) operation
    return sorted_list


def time_it(
    func: Callable, *args: Any, repetitions: int = 1, **kwargs: Any
) -> float:
    """
    Measures the average execution time of a function.

    Args:
        func: The function to time.
        *args: Positional arguments to pass to the function.
        repetitions: The number of times to run the function.
        **kwargs: Keyword arguments to pass to the function.

    Returns:
        The average execution time in seconds.
    """
    if not isinstance(repetitions, int) or repetitions <= 0:
        raise ValueError("Repetitions must be a positive integer.")

    total_time = 0.0
    for _ in range(repetitions):
        start = time.perf_counter()
        func(*args, **kwargs)
        end = time.perf_counter()
        total_time += (end - start)
    
    return total_time / repetitions


def simple_interest(principal: float, time_years: float, rate_percent: float) -> float:
    """Calculates simple interest."""
    return (principal * time_years * rate_percent) / 100


def find_primes_in_range(start: int, end: int) -> List[int]:
    """
    Finds all prime numbers within a given interval (inclusive of start).

    Args:
        start: The beginning of the range.
        end: The end of the range.

    Returns:
        A list of prime numbers found in the range.
    """
    primes = []
    for num in range(start, end):
        if num > 1:
            # Check for factors from 2 up to the square root of the number
            for i in range(2, int(math.sqrt(num)) + 1):
                if (num % i) == 0:
                    break  # It's not a prime
            else:
                primes.append(num)  # Loop finished without finding a factor
    return primes


def make_word_counter() -> Callable[[str], int]:
    """
    Creates a counter function using a closure. The returned function keeps
    track of word counts across calls.

    Returns:
        A function that takes a word and returns its updated count.
    """
    counts: Dict[str, int] = {}

    def counter(word: str) -> int:
        """Increments and returns the count for a given word."""
        counts[word] = counts.get(word, 0) + 1
        return counts[word]

    return counter


def is_palindrome(text: str) -> bool:
    """
    Checks if a string is a palindrome in a Pythonic way.
    A palindrome reads the same forwards and backwards.

    Args:
        text: The string to check.

    Returns:
        True if the string is a palindrome, False otherwise.
    """
    # Pre-process the string to be case-insensitive and ignore non-alphanumeric chars
    processed_text = ''.join(char.lower() for char in text if char.isalnum())
    return processed_text == processed_text[::-1]


def find_urls_in_string(text: str) -> List[str]:
    """
    Finds all URLs in a given string using a regular expression.

    Args:
        text: The string to search for URLs.

    Returns:
        A list of URLs found in the string.
    """
    # A common regex for finding URLs.
    url_regex = (
        r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)"
        r"(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+"
        r"(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    )
    urls = re.findall(url_regex, text)
    # The regex returns tuples due to capturing groups; we only need the first element.
    return [match[0] for match in urls]


def find_n_largest_elements(data: List[int], n: int) -> List[int]:
    """
    Finds the N largest elements from a list.
    Note: For very large lists and small N, `heapq.nlargest(n, data)` is more efficient.

    Args:
        data: The list of numbers.
        n: The number of largest elements to find.

    Returns:
        A list containing the N largest elements, sorted.
    """
    if n <= 0:
        return []
    return sorted(data)[-n:]


def is_close_float(a: float, b: float, rel_tol: float = 1e-9, abs_tol: float = 1e-9) -> bool:
    """
    Tests the equality of two floating-point numbers within a tolerance.
    This is equivalent to `math.isclose()` available in Python 3.5+.

    Args:
        a: The first float.
        b: The second float.
        rel_tol: The relative tolerance.
        abs_tol: The absolute tolerance.

    Returns:
        True if the numbers are close enough, False otherwise.
    """
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def is_fibonacci_number(n: int) -> bool:
    """
    Checks if a number is a Fibonacci number using a mathematical property.
    A number `n` is a Fibonacci number if and only if (5*n^2 + 4) or
    (5*n^2 - 4) is a perfect square.

    Args:
        n: A non-negative integer.

    Returns:
        True if n is a Fibonacci number, False otherwise.
    """
    if not isinstance(n, int) or n < 0:
        raise TypeError("Input must be a non-negative integer.")

    def is_perfect_square(x: int) -> bool:
        if x < 0:
            return False
        sqrt_x = int(math.sqrt(x))
        return sqrt_x * sqrt_x == x

    return is_perfect_square(5 * n * n + 4) or is_perfect_square(5 * n * n - 4)


def make_fibonacci_generator() -> Callable[[], int]:
    """
    Returns a function that generates Fibonacci numbers sequentially on each call.

    Returns:
        A closure function that acts as a Fibonacci number generator.
    """
    a, b = 0, 1

    def get_next_fibonacci() -> int:
        nonlocal a, b
        result = a
        a, b = b, a + b
        return result

    return get_next_fibonacci
    
# A more Pythonic way using a generator function with `yield`
def fibonacci_sequence_generator():
    """Yields an infinite sequence of Fibonacci numbers."""
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b


def factorial(n: int) -> int:
    """
    Calculates the factorial of a non-negative integer iteratively.

    Args:
        n: The number to calculate the factorial of.

    Returns:
        The factorial of n.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0:
        return 1
    
    # Python 3.8+ has math.prod for a more concise version:
    # return math.prod(range(1, n + 1))
    
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def is_leap_year(year: int) -> bool:
    """
    Determines if a given year is a leap year.

    A year is a leap year if it is divisible by 4, except for end-of-century
    years, which must be divisible by 400.

    Args:
        year: The year to check.

    Returns:
        True if it's a leap year, False otherwise.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


def compute_gcd(x: int, y: int) -> int:
    """
    Computes the greatest common divisor (GCD) of two integers using the
    Euclidean algorithm. `math.gcd` is available in Python 3.5+.
    """
    while y:
        x, y = y, x % y
    return abs(x)


def compute_lcm(x: int, y: int) -> int:
    """
    Computes the least common multiple (LCM) of two integers.
    Uses the formula: lcm(x, y) = (|x * y|) / gcd(x, y)
    """
    if x == 0 or y == 0:
        return 0
    return abs(x * y) // compute_gcd(x, y)


def remove_punctuation(text: str, punctuation: str = string.punctuation) -> str:
    """
    Removes all punctuation characters from a string using `str.translate`.

    Args:
        text: The input string.
        punctuation: A string of punctuation characters to remove.

    Returns:
        The string with punctuation removed.
    """
    translation_table = str.maketrans('', '', punctuation)
    return text.translate(translation_table)


def get_file_hash(filepath: str, algorithm: str = 'sha1') -> str:
    """
    Calculates the hash of a file. Reads the file in chunks to handle large files.

    Args:
        filepath: The path to the file.
        algorithm: The hashing algorithm to use (e.g., 'sha1', 'md5', 'sha256').

    Returns:
        The hexadecimal hash digest of the file.
    """
    hasher = hashlib.new(algorithm)
    chunk_size = 4096  # 4 KB chunks
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()


def solve_quadratic_equation(a: float, b: float, c: float) -> Tuple[complex, complex]:
    """
    Solves a quadratic equation ax^2 + bx + c = 0.

    Returns:
        A tuple containing the two roots (which may be complex).
    """
    # Calculate the discriminant
    delta = (b**2) - (4 * a * c)
    
    # Find the two solutions
    sol1 = (-b - cmath.sqrt(delta)) / (2 * a)
    sol2 = (-b + cmath.sqrt(delta)) / (2 * a)
    return sol1, sol2


def is_armstrong_number(num: int) -> bool:
    """
    Checks if a number is an Armstrong number.
    An Armstrong number is a number that is equal to the sum of its own digits
    each raised to the power of the number of digits. e.g., 153 = 1^3 + 5^3 + 3^3.

    Args:
        num: The integer to check.

    Returns:
        True if it is an Armstrong number, False otherwise.
    """
    if not isinstance(num, int) or num < 0:
        return False
        
    s_num = str(num)
    order = len(s_num)
    
    total = sum(int(digit) ** order for digit in s_num)
    
    return num == total


def sum_natural_numbers(n: int) -> int:
    """
    Calculates the sum of natural numbers up to n.
    Uses the efficient mathematical formula: n * (n + 1) / 2.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    return n * (n + 1) // 2


def create_shuffled_deck() -> List[Tuple[Union[int, str], str]]:
    """Creates a standard 52-card deck and shuffles it."""
    ranks = list(range(2, 11)) + ['J', 'Q', 'K', 'A']
    suits = ['Spades', 'Hearts', 'Diamonds', 'Clubs']
    deck = list(itertools.product(ranks, suits))
    random.shuffle(deck)
    return deck


def decimal_to_binary(n: int) -> str:
    """
    Converts a decimal integer to its binary string representation.
    This is a wrapper around the built-in `bin()` function.
    """
    if n < 0:
        return "-" + bin(n)[3:]
    return bin(n)[2:]


def are_anagrams(s1: str, s2: str) -> bool:
    """
    Checks if two strings are anagrams of each other.
    Anagrams are words formed by rearranging the letters of another.
    """
    # A more robust check would ignore case and non-alphanumeric characters.
    return sorted(s1.lower()) == sorted(s2.lower())


def remove_duplicates_preserve_order(data: List[Any]) -> List[Any]:
    """
    Removes duplicate items from a list while preserving the original order.
    Uses the `dict.fromkeys()` method, which is efficient and order-preserving (in Python 3.7+).
    """
    return list(dict.fromkeys(data))


# =============================================================================
# 4. Classes and Object-Oriented Programming
# =============================================================================

class Circle:
    """A class representing a circle."""
    def __init__(self, radius: float):
        """
        Initializes a Circle object.

        Args:
            radius: The radius of the circle. Must be non-negative.
        """
        if radius < 0:
            raise ValueError("Radius cannot be negative.")
        self.radius = radius

    def area(self) -> float:
        """Calculates the area of the circle."""
        return math.pi * self.radius ** 2

    def circumference(self) -> float:
        """Calculates the circumference of the circle."""
        return 2 * math.pi * self.radius


class Queue:
    """
    A thread-safe, memory-efficient, fixed-size FIFO (First-In, First-Out) queue.
    This implementation is a wrapper around `collections.deque`.
    """

    def __init__(self, max_size: int = 10):
        """
        Initializes the queue with a maximum size.

        Args:
            max_size: Maximum number of items the queue can hold. Defaults to 10.
        """
        self._queue = deque(maxlen=max_size)

    def enqueue(self, item: Any) -> None:
        """
        Adds an item to the tail of the queue.
        If the queue is full, the item at the head is automatically removed.
        """
        self._queue.append(item)

    def dequeue(self) -> Any:
        """
        Removes and returns the item from the head of the queue.

        Raises:
            IndexError: If the queue is empty.
        """
        # Use popleft() for FIFO behavior (Queue).
        # The original code used pop(), which is LIFO (Stack).
        return self._queue.popleft()
        
    def __len__(self) -> int:
        """Returns the current number of items in the queue."""
        return len(self._queue)
        
    def __str__(self) -> str:
        """Returns a string representation of the queue."""
        return f"Queue(items={list(self._queue)}, max_size={self._queue.maxlen})"


# =============================================================================
# 5. Main Execution Block
# =============================================================================

if __name__ == "__main__":
    # This block runs only when the script is executed directly.
    # It's the perfect place to demonstrate the functions and classes defined above.

    print("===== Running Python Code Examples =====")

    # --- Core Concepts ---
    demonstrate_random_choice()
    demonstrate_generator_lazy_evaluation()
    demonstrate_args(0, 1, 'a', 'b', 'c')
    demonstrate_kwargs(a=1, b=2, c=3, d=4)
    demonstrate_iterable_unpacking()
    demonstrate_for_else_loop()
    demonstrate_while_else_loop()
    demonstrate_string_interning()
    demonstrate_disassembler()

    # --- Functions and Algorithms ---
    print("\n--- Function Demonstrations ---")
    greet("World")
    print(f"Absolute value of -4 is {absolute_value(-4)}")
    print(f"10th Catalan number is: {catalan(9)}")
    print(f"Binomial Coefficient C(5, 2) is: {binomial_coefficient(5, 2)}")
    
    lis_arr = [10, 22, 9, 33, 21, 50, 41, 60]
    print(f"Length of LIS for {lis_arr} is: {find_longest_increasing_subsequence(lis_arr)}")
    
    print(f"9th Fibonacci number is: {fibonacci_recursive(9)}")
    print(f"1 to 100 KM/H is {kmh_to_mph(100):.2f} MPH")
    print(f"Smallest of (100, 200, 300) is: {min(100, 200, 300)}")
    
    raw_list = [-5, -23, 5, 0, 23, -6, 23, 67]
    print(f"Custom sorted list: {custom_sort(raw_list)}")
    print(f"Built-in sorted list: {sorted(raw_list)}")

    print(f"Primes between 11 and 25: {find_primes_in_range(11, 25)}")
    
    word_counter = make_word_counter()
    word_counter("hello")
    word_counter("world")
    print(f"Count of 'hello': {word_counter('hello')}")
    
    print(f"'malayalam' is a palindrome: {is_palindrome('malayalam')}")
    print(f"'A man, a plan, a canal: Panama' is a palindrome: {is_palindrome('A man, a plan, a canal: Panama')}")

    test_string_for_urls = "Check out my website at http://example.com and also www.anotherexample.org."
    print(f"URLs found: {find_urls_in_string(test_string_for_urls)}")

    num_list = [1000, 298, 3579, 100, 200, -45, 900]
    print(f"4 largest elements: {find_n_largest_elements(num_list, 4)}")
    
    nums1 = [1, 2, 3]
    nums2 = [4, 5, 6]
    summed_list = list(map(lambda x, y: x + y, nums1, nums2))
    print(f"Sum of {nums1} and {nums2} is {summed_list}")
    
    print(f"Is 8 a Fibonacci number? {is_fibonacci_number(8)}")
    print(f"Is 9 a Fibonacci number? {is_fibonacci_number(9)}")
    
    sol1, sol2 = solve_quadratic_equation(1, 5, 6)
    print(f"Solutions for x^2 + 5x + 6 = 0 are: {sol1} and {sol2}")
    
    print(f"1634 is an Armstrong number: {is_armstrong_number(1634)}")
    print(f"150 is an Armstrong number: {is_armstrong_number(150)}")

    print(f"Shuffled deck (first 5 cards): {create_shuffled_deck()[:5]}")

    # --- Class Demonstrations ---
    print("\n--- Class Demonstrations ---")
    my_circle = Circle(radius=10)
    print(f"Circle with radius 10 has area: {my_circle.area():.2f}")
    
    my_queue = Queue(max_size=3)
    my_queue.enqueue("A")
    my_queue.enqueue("B")
    my_queue.enqueue("C")
    print(f"Initial queue: {my_queue}")
    
    my_queue.enqueue("D") # This will push "A" out
    print(f"Queue after adding 'D': {my_queue}")
    
    dequeued_item = my_queue.dequeue()
    print(f"Dequeued item: {dequeued_item}. Queue is now: {my_queue}")