# ==============================================================================
# Refactored Python Code Examples
#
# This script contains a collection of refactored Python functions and code
# snippets, demonstrating best practices for readability, reusability, and
# adherence to Python standards (PEP 8).
#
# To install required external libraries for web scraping:
# pip install requests beautifulsoup4
# ==============================================================================

# --- Standard Library Imports ---
import collections
import gc
import heapq
import itertools
import math
import random
import re
import string
import time
from functools import reduce, wraps
from typing import (Any, Callable, Deque, Dict, Iterable, List, Optional, Set,
                    Tuple, Union)

# --- Third-Party Imports (for specific functions) ---
# Note: These are optional. The script runs without them, but the
#       `get_profanity_list_from_url` function will fail.
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Warning: 'requests' and 'beautifulsoup4' not found.")
    print("Run 'pip install requests beautifulsoup4' to use 'get_profanity_list_from_url'.")
    requests = None
    BeautifulSoup = None


# ==============================================================================
# Section 1: String Manipulation
# ==============================================================================

def get_unique_words(sentence: str) -> Set[str]:
    """
    Returns a set of unique words from a given sentence.

    Args:
        sentence: The input string.

    Returns:
        A set containing the unique words.
    """
    return set(sentence.lower().split())


def get_punctuations_from_string(text: str) -> Set[str]:
    """
    Finds and returns all punctuation characters used in a string.

    Args:
        text: The input string.

    Returns:
        A set of punctuation characters found in the text.
    """
    return {char for char in text if char in string.punctuation}


def reverse_word_order(sentence: str) -> str:
    """
    Reverses the order of words in a sentence.

    Args:
        sentence: The input string.

    Returns:
        A new string with the words in reverse order.
    """
    return ' '.join(reversed(sentence.split()))


def replace_words_with_length(sentence: str) -> str:
    """
    Replaces each word in a sentence with its length.

    Args:
        sentence: The input string.

    Returns:
        A string where each word is replaced by its character count.
    """
    lengths = [str(len(word)) for word in sentence.split()]
    return ' '.join(lengths)


def filter_words_by_length(sentence: str, min_length: int = 0, max_length: Optional[int] = None) -> List[str]:
    """
    Filters words in a sentence based on their length.

    Args:
        sentence: The input string.
        min_length: The minimum length of words to keep.
        max_length: The maximum length of words to keep. If None, no upper
                    limit is applied.

    Returns:
        A list of words that meet the length criteria.
    """
    words = sentence.split()
    if max_length is not None:
        return [word for word in words if min_length <= len(word) <= max_length]
    return [word for word in words if len(word) >= min_length]


def strip_vowels(text: str) -> str:
    """
    Removes all vowels (case-insensitive) from a string.

    Args:
        text: The input string.

    Returns:
        The string with all vowels removed.
    """
    vowels = "aeiouAEIOU"
    return "".join(char for char in text if char not in vowels)


def repeat_string(text: str, n: int) -> str:
    """
    Repeats a string n times.

    Args:
        text: The string to repeat.
        n: The number of times to repeat the string.

    Returns:
        The repeated string.
    """
    return text * n


def clean_string(text: str) -> str:
    """
    Removes all non-alphanumeric characters from a string.

    Args:
        text: The input string.

    Returns:
        A cleaned string containing only letters and numbers.
    """
    return re.sub(r'[^A-Za-z0-9]+', '', text)


def replace_substring(text: str, old: str, new: str) -> str:
    """
    Replaces all occurrences of a substring with a new one.

    Args:
        text: The original string.
        old: The substring to replace.
        new: The substring to replace with.

    Returns:
        The modified string.
    """
    return text.replace(old, new)


# ==============================================================================
# Section 2: List, Tuple, and Iterable Operations
# ==============================================================================

def make_negatives_zero(numbers: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Replaces all negative numbers in a list with zero.

    Args:
        numbers: A list of numbers.

    Returns:
        A new list with negative numbers replaced by 0.
    """
    return [max(0, num) for num in numbers]


def abs_value_list(numbers: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Converts all numbers in a list to their absolute (positive) values.

    Args:
        numbers: A list of numbers.

    Returns:
        A new list with all numbers being positive.
    """
    return [abs(num) for num in numbers]


def make_all_negative(numbers: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Converts all numbers in a list to their negative values.

    Args:
        numbers: A list of numbers.

    Returns:
        A new list with all numbers being negative.
    """
    return [-abs(num) for num in numbers]


def create_index_dict_from_list(items: List[Any]) -> Dict[int, Any]:
    """
    Converts a list to a dictionary with keys as indices and values as items.

    Args:
        items: The input list.

    Returns:
        A dictionary mapping index to item.
    """
    return dict(enumerate(items))


def add_from_iterables_if(iter1: Iterable[int], iter2: Iterable[int]) -> List[int]:
    """
    Adds elements from two iterables if the element from the first is even
    and the element from the second is odd.

    Args:
        iter1: The first iterable of integers.
        iter2: The second iterable of integers.

    Returns:
        A list of sums where the condition is met.
    """
    return [a + b for a, b in zip(iter1, iter2) if a % 2 == 0 and b % 2 != 0]


def get_top_element(numbers: List[Union[int, float]]) -> Union[int, float]:
    """
    Sorts a list of numbers and returns the largest element.

    Args:
        numbers: A list of numbers.

    Returns:
        The largest number in the list.
    """
    if not numbers:
        raise ValueError("Input list cannot be empty")
    return max(numbers)


def merge_iterables_pairwise(iter1: Iterable, iter2: Iterable) -> List[Tuple]:
    """
    Merges two iterables into a list of tuples (pairs).

    Args:
        iter1: The first iterable.
        iter2: The second iterable.

    Returns:
        A list of tuples, where each tuple contains elements from both iterables.
    """
    return list(zip(iter1, iter2))


def append_lists(list1: List[Any], list2: List[Any]) -> List[Any]:
    """
    Appends the elements of a second list to a first list, returning a new list.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        A new list containing all elements from both lists.
    """
    return list1 + list2


def elementwise_list_operation(list1: List[Union[int, float]],
                               list2: List[Union[int, float]],
                               op: Callable) -> List[Union[int, float]]:
    """
    Performs an element-wise operation on two lists.

    Args:
        list1: The first list of numbers.
        list2: The second list of numbers.
        op: A function that takes two numbers and returns a number (e.g., lambda x, y: x + y).

    Returns:
        A new list with the results of the operation.
    """
    return [op(i, j) for i, j in zip(list1, list2)]


def has_no_duplicates(items: List[Any]) -> bool:
    """
    Checks if a list contains any duplicate items.

    Args:
        items: The list to check.

    Returns:
        True if the list has no duplicates, False otherwise.
    """
    return len(items) == len(set(items))


# ==============================================================================
# Section 3: Mathematical Functions
# ==============================================================================

def get_factors(number: int) -> List[int]:
    """
    Returns a list of all factors of a given integer.

    Args:
        number: A positive integer.

    Returns:
        A list of its factors.
    """
    if number <= 0:
        return []
    return [i for i in range(1, number + 1) if number % i == 0]


def is_product_even(num1: int, num2: int) -> bool:
    """
    Checks if the product of two numbers is even.

    Args:
        num1: The first number.
        num2: The second number.

    Returns:
        True if the product is even, False otherwise.
    """
    return (num1 * num2) % 2 == 0


def is_sum_even(num1: int, num2: int) -> bool:
    """
    Checks if the sum of two numbers is even.

    Args:
        num1: The first number.
        num2: The second number.

    Returns:
        True if the sum is even, False otherwise.
    """
    return (num1 + num2) % 2 == 0


def calculate_factorial(n: int) -> int:
    """
    Calculates the factorial of a non-negative integer using math.factorial.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial of n.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    return math.factorial(n)


def calculate_factorial_recursive(n: int) -> int:
    """
    Calculates the factorial of a non-negative integer using recursion.
    Note: Less efficient than the iterative or math.factorial version.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial of n.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0 or n == 1:
        return 1
    else:
        return n * calculate_factorial_recursive(n - 1)


def sum_first_n_numbers(n: int) -> int:
    """
    Calculates the sum of the first n positive integers.

    Args:
        n: A positive integer.

    Returns:
        The sum of integers from 1 to n.
    """
    if n < 0:
        return 0
    # Formula for sum of an arithmetic series: n * (n + 1) / 2
    # More efficient than a loop for large n.
    return n * (n + 1) // 2


def fibonacci_recursive(n: int) -> int:
    """
    Generates the nth Fibonacci number using recursion.
    Warning: Highly inefficient for n > 35 due to repeated calculations.

    Args:
        n: The index in the Fibonacci sequence (starting from 0).

    Returns:
        The nth Fibonacci number.
    """
    if n < 0:
        raise ValueError("Fibonacci index cannot be negative.")
    if n <= 1:
        return n
    return fibonacci_recursive(n - 1) + fibonacci_recursive(n - 2)


def derivative(func: Callable[[float], float], x: float, h: float = 1e-6) -> float:
    """
    Calculates the numerical derivative of a function at a point x.

    Args:
        func: The function to differentiate (e.g., math.sin).
        x: The point at which to calculate the derivative.
        h: A small step size for the calculation.

    Returns:
        The approximate derivative of the function at x.
    """
    return (func(x + h) - func(x - h)) / (2 * h)


def relu(x: Union[int, float]) -> Union[int, float]:
    """
    Applies the Rectified Linear Unit (ReLU) activation function.

    Args:
        x: A number.

    Returns:
        x if x > 0, else 0.
    """
    return max(0, x)


def relu_derivative(x: Union[int, float]) -> int:
    """
    Calculates the derivative of the ReLU function.

    Args:
        x: A number.

    Returns:
        1 if x > 0, else 0.
    """
    return 1 if x > 0 else 0


# ==============================================================================
# Section 4: Decorators and Advanced Functions
# ==============================================================================

def function_call_counter(func: Callable) -> Callable:
    """
    A decorator that counts and prints the number of times a function is called.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.call_count += 1
        print(f"Function '{func.__name__}' has been called {wrapper.call_count} times.")
        return func(*args, **kwargs)
    wrapper.call_count = 0
    return wrapper


def timing_decorator(func: Callable) -> Callable:
    """
    A decorator that measures and prints the execution time of a function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(f"Function '{func.__name__}' took {elapsed_time:.6f} seconds to execute.")
        return result
    return wrapper


def time_it(func: Callable, *args, repetitions: int = 1, **kwargs) -> float:
    """
    Calculates the average execution time of a function over several repetitions.

    Args:
        func: The function to time.
        repetitions: The number of times to run the function.
        *args, **kwargs: Arguments to pass to the function.

    Returns:
        The average execution time in seconds.
    """
    total_time = 0.0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
    return total_time / repetitions


# ==============================================================================
# Section 5: Generators
# ==============================================================================

def infinite_squares_generator() -> Iterable[int]:
    """
    A generator that yields an infinite sequence of square numbers (1, 4, 9, ...).
    """
    num = 1
    while True:
        yield num * num
        num += 1


def fibonacci_generator(limit: int) -> Iterable[int]:
    """
    A generator that yields Fibonacci numbers up to a given limit.

    Args:
        limit: The exclusive upper bound for the Fibonacci numbers.

    Yields:
        Fibonacci numbers one by one.
    """
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b


# ==============================================================================
# Section 6: I/O and Web-related Functions
# ==============================================================================

def get_profanity_list_from_url(url: str) -> List[str]:
    """
    Scrapes a list of profanity words from a specific GitHub URL.
    NOTE: This is fragile and depends on the page's HTML structure.

    Args:
        url: The URL to the list.txt file on GitHub.

    Returns:
        A list of profanity words.
    """
    if not requests or not BeautifulSoup:
        print("Error: 'requests' and 'beautifulsoup4' are required for this function.")
        return []

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, "html.parser")
        # Find the table containing the file content
        table = soup.find('table')
        if not table:
            return []
        # Extract text from each table cell in the file content rows
        return [td.text for tr in table.find_all('tr') for td in tr.find_all('td', class_='blob-code')]
    except requests.RequestException as e:
        print(f"Error fetching URL: {e}")
        return []


# ==============================================================================
# Main execution block
# ==============================================================================

def main():
    """Main function to demonstrate the refactored code."""
    print("--- Demonstrating Refactored Functions ---")

    # --- String Manipulation ---
    sentence = "The quick brown fox jumps over the lazy dog"
    print(f"\nOriginal sentence: '{sentence}'")
    print(f"Unique words: {get_unique_words(sentence)}")
    print(f"Reversed word order: '{reverse_word_order(sentence)}'")
    print(f"Words replaced with length: '{replace_words_with_length(sentence)}'")
    print(f"Words with min length 4: {filter_words_by_length(sentence, min_length=4)}")
    print(f"Sentence with vowels stripped: '{strip_vowels(sentence)}'")

    # --- List, Tuple, and Iterable Operations ---
    my_list = [1, -2, 3, -4, 5, 5]
    print(f"\nOriginal list: {my_list}")
    print(f"Negatives made zero: {make_negatives_zero(my_list)}")
    print(f"Absolute values: {abs_value_list(my_list)}")
    print(f"Does the list have duplicates? {not has_no_duplicates(my_list)}")
    
    list1 = [2, 4, 6]
    list2 = [1, 3, 5]
    print(f"Element-wise sum of {list1} and {list2}: {elementwise_list_operation(list1, list2, lambda x, y: x + y)}")

    # --- Mathematical Functions ---
    num = 12
    print(f"\nFactors of {num}: {get_factors(num)}")
    print(f"Factorial of 5: {calculate_factorial(5)}")
    print(f"Sum of first 10 numbers: {sum_first_n_numbers(10)}")
    print(f"Is product of 3 and 4 even? {is_product_even(3, 4)}")
    print(f"Is sum of 3 and 4 even? {is_sum_even(3, 4)}")

    # --- Decorators ---
    @timing_decorator
    @function_call_counter
    def example_function(a, b):
        """An example function to test decorators."""
        time.sleep(0.1) # Simulate work
        return a + b

    print("\n--- Testing Decorators ---")
    example_function(10, 20)
    example_function(30, 40)
    
    # --- Generators ---
    print("\n--- Testing Generators ---")
    print("First 5 square numbers:")
    sq_gen = infinite_squares_generator()
    for _ in range(5):
        print(next(sq_gen), end=" ")
    print("\nFibonacci numbers less than 50:")
    for num in fibonacci_generator(50):
        print(num, end=" ")
    print()


if __name__ == "__main__":
    main()
