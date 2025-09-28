# itertools_recipes.py

"""
A collection of highly reusable functions inspired by the Python itertools
documentation recipes. These functions provide powerful and memory-efficient
tools for working with iterables.
"""

from itertools import (
    chain,
    combinations,
    cycle,
    filterfalse,
    groupby,
    islice,
    repeat,
    starmap,
    tee,
    zip_longest,
)
import operator
import random
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    TypeVar,
    Tuple,
)

# A generic type variable to represent the type of elements in an iterable.
T = TypeVar("T")


def repeat_func(
    func: Callable[..., T], times: Optional[int] = None, *args: Any
) -> Iterator[T]:
    """
    Repeat calls to a function with specified arguments.

    :param func: The function to be called repeatedly.
    :param times: The number of times to repeat the call. If None, repeats indefinitely.
    :param args: The arguments to pass to the function.
    :return: An iterator yielding the results of the function calls.

    >>> from operator import add
    >>> list(repeat_func(add, times=3, *[10, 20]))
    [30, 30, 30]
    """
    if times is None:
        return starmap(func, repeat(args))
    return starmap(func, repeat(args, times))


def pairwise(iterable: Iterable[T]) -> Iterator[Tuple[T, T]]:
    """
    Return successive overlapping pairs from an iterable.
    s -> (s0,s1), (s1,s2), (s2, s3), ...

    :param iterable: The source iterable.
    :return: An iterator of overlapping pairs.

    >>> list(pairwise('ABCDEFG'))
    [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'), ('E', 'F'), ('F', 'G')]
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def grouper(
    iterable: Iterable[T], n: int, fillvalue: Optional[Any] = None
) -> Iterator[Tuple[T, ...]]:
    """
    Collect data into fixed-length chunks or blocks.

    :param iterable: The source iterable.
    :param n: The size of each chunk.
    :param fillvalue: The value to use for padding the last chunk if it's short.
    :return: An iterator of tuples, where each tuple is a chunk.

    >>> list(grouper('ABCDEFG', 3, 'x'))
    [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]
    """
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def roundrobin(*iterables: Iterable[T]) -> Iterator[T]:
    """
    Yields items from several iterables in a round-robin fashion.

    :param iterables: One or more iterables.
    :return: An iterator that yields elements from the input iterables in turn.

    >>> list(roundrobin('ABC', 'D', 'EF'))
    ['A', 'D', 'E', 'B', 'F', 'C']
    """
    num_active = len(iterables)
    nexts = cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next_item in nexts:
                yield next_item()
        except StopIteration:
            num_active -= 1
            nexts = cycle(islice(nexts, num_active))


def partition(
    pred: Callable[[T], bool], iterable: Iterable[T]
) -> Tuple[Iterator[T], Iterator[T]]:
    """
    Use a predicate to partition entries into false entries and true entries.

    :param pred: A function that returns True or False for an item.
    :param iterable: The source iterable.
    :return: A tuple containing two iterators: one for items where pred is False,
             and one for items where pred is True.

    >>> is_odd = lambda x: x % 2 != 0
    >>> false_items, true_items = partition(is_odd, range(10))
    >>> list(false_items)
    [0, 2, 4, 6, 8]
    >>> list(true_items)
    [1, 3, 5, 7, 9]
    """
    t1, t2 = tee(iterable)
    return filterfalse(pred, t1), filter(pred, t2)


def powerset(iterable: Iterable[T]) -> Iterator[Tuple[T, ...]]:
    """
    Generates the powerset of an iterable.
    powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)

    :param iterable: The source iterable.
    :return: An iterator yielding all subsets of the iterable.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


def unique_everseen(
    iterable: Iterable[T], key: Optional[Callable[[T], Any]] = None
) -> Iterator[T]:
    """
    Yield unique elements, preserving order. Remembers all elements ever seen.

    :param iterable: The source iterable.
    :param key: A function to extract a comparison key from each element.
    :return: An iterator of unique elements.

    >>> list(unique_everseen('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D']
    >>> list(unique_everseen('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'D']
    """
    seen: Set = set()
    seen_add = seen.add
    if key is None:
        for element in filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def unique_justseen(
    iterable: Iterable[T], key: Optional[Callable[[T], Any]] = None
) -> Iterator[T]:
    """
    Yield unique elements, preserving order. Remembers only the element just seen.

    :param iterable: The source iterable.
    :param key: A function to extract a comparison key from each element.
    :return: An iterator of unique elements based on consecutive grouping.

    >>> list(unique_justseen('AAAABBBCCDAABBB'))
    ['A', 'B', 'C', 'D', 'A', 'B']
    >>> list(unique_justseen('ABBCcAD', str.lower))
    ['A', 'B', 'C', 'A', 'D']
    """
    return map(next, map(operator.itemgetter(1), groupby(iterable, key)))


def iter_except(
    func: Callable[[], T],
    exception: type,
    first: Optional[Callable[[], T]] = None,
) -> Iterator[T]:
    """
    Converts a call-until-exception interface to an iterator interface.

    Acts like `iter(func, sentinel)` but uses an exception to end the loop.

    :param func: The function to call repeatedly.
    :param exception: The exception type that signals the end of iteration.
    :param first: An optional function to call once before the loop begins.
    :return: An iterator that yields values from `func` until `exception` is raised.
    """
    try:
        if first:
            yield first()
        while True:
            yield func()
    except exception:
        return


def random_product(*args: Iterable[T], repeat: int = 1) -> Tuple[T, ...]:
    """
    Return a random element from the cartesian product of iterables.

    Equivalent to `random.choice(list(itertools.product(*args, **kwds)))`.

    :param args: One or more iterables.
    :param repeat: The number of times to repeat the iterables.
    :return: A tuple representing a random choice from the product.
    """
    pools: List[Tuple[T, ...]] = [tuple(pool) for pool in args] * repeat
    return tuple(random.choice(pool) for pool in pools)


def random_permutation(iterable: Iterable[T], r: Optional[int] = None) -> Tuple[T, ...]:
    """
    Return a random r-length permutation of elements from the iterable.

    :param iterable: The source iterable.
    :param r: The length of the permutation. Defaults to the length of the iterable.
    :return: A random permutation tuple.
    """
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def random_combination(iterable: Iterable[T], r: int) -> Tuple[T, ...]:
    """
    Return a random r-length combination of elements from the iterable.

    :param iterable: The source iterable.
    :param r: The length of the combination.
    :return: A random combination tuple.
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        raise ValueError("r must be less than or equal to the length of the iterable")
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_combination_with_replacement(iterable: Iterable[T], r: int) -> Tuple[T, ...]:
    """
    Return a random r-length combination with replacement of elements.

    :param iterable: The source iterable.
    :param r: The length of the combination.
    :return: A random combination with replacement tuple.
    """
    pool = tuple(iterable)
    n = len(pool)
    if n == 0 and r > 0:
        raise ValueError("Cannot choose from an empty sequence")
    indices = sorted(random.choices(range(n), k=r))
    return tuple(pool[i] for i in indices)



# math_helpers.py

"""
A collection of mathematical and numerical helper functions.
Includes numerical search algorithms using `bisect` and common
mathematical operations.
"""

import math
from bisect import bisect, bisect_left, bisect_right
from typing import List, Sequence, TypeVar, Union

# Define a type for number-like objects for type hints
Numeric = TypeVar("Numeric", int, float)
Number = Union[int, float]


# --- Bisect-based Search Functions ---

def index(a: Sequence[T], x: T) -> int:
    """
    Locate the leftmost value exactly equal to x in a sorted sequence.

    :param a: The sorted sequence.
    :param x: The value to find.
    :return: The index of the first occurrence of x.
    :raises ValueError: If x is not found in a.
    """
    i = bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError(f"{x} is not in list")


def find_lt(a: Sequence[Numeric], x: Numeric) -> Numeric:
    """Find the rightmost value in a sorted sequence less than x."""
    i = bisect_left(a, x)
    if i:
        return a[i - 1]
    raise ValueError(f"No value found less than {x}")


def find_le(a: Sequence[Numeric], x: Numeric) -> Numeric:
    """Find the rightmost value in a sorted sequence less than or equal to x."""
    i = bisect_right(a, x)
    if i:
        return a[i - 1]
    raise ValueError(f"No value found less than or equal to {x}")


def find_gt(a: Sequence[Numeric], x: Numeric) -> Numeric:
    """Find the leftmost value in a sorted sequence greater than x."""
    i = bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError(f"No value found greater than {x}")


def find_ge(a: Sequence[Numeric], x: Numeric) -> Numeric:
    """Find the leftmost value in a sorted sequence greater than or equal to x."""
    i = bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError(f"No value found greater than or equal to {x}")


def grade(score: Number, breakpoints: List[Number] = [60, 70, 80, 90], grades: str = 'FDCBA') -> str:
    """
    Assigns a letter grade based on a score using a sorted list of breakpoints.

    :param score: The score to be graded.
    :param breakpoints: A sorted list of score boundaries.
    :param grades: A string of grades corresponding to the breakpoints.
    :return: The letter grade.
    
    >>> grade(85)
    'B'
    >>> grade(95)
    'A'
    """
    i = bisect(breakpoints, score)
    return grades[i]


# --- Activation and Mathematical Functions ---

def relu(x: Number) -> Number:
    """Rectified Linear Unit (ReLU) activation function."""
    return max(0, x)


def derivative_relu(x: Number) -> int:
    """Derivative of the ReLU function."""
    return 1 if x > 0 else 0


def factorial(n: int) -> int:
    """Calculates the factorial of a non-negative integer recursively."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0 or n == 1:
        return 1
    else:
        # Note: For very large n, an iterative approach or math.factorial is more efficient
        return n * factorial(n - 1)


def fibonacci(n: int) -> int:
    """
    Generates the nth Fibonacci number (starting with F(0)=0, F(1)=1).
    
    Note: This recursive implementation is inefficient for large n due to
    re-computation. An iterative or memoized version is preferred for performance.
    """
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    if n == 0:
        return 0
    elif n == 1:
        return 1
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)

# --- Trigonometric and Exponential Functions (Wrappers around math module) ---

def sin(x: Number) -> float:
    """Returns the sine of x."""
    return math.sin(x)

def derivative_sin(x: Number) -> float:
    """Returns the derivative of sine (cosine) of x."""
    return math.cos(x)

def cos(x: Number) -> float:
    """Returns the cosine of x."""
    return math.cos(x)

def derivative_cos(x: Number) -> float:
    """Returns the derivative of cosine (-sine) of x."""
    return -math.sin(x)

def tan(x: Number) -> float:
    """Returns the tangent of x."""
    return math.tan(x)

def derivative_tan(x: Number) -> float:
    """Returns the derivative of tangent (sec^2) of x."""
    return 1 / (math.cos(x) ** 2)

def exp(x: Number) -> float:
    """Returns the exponential of x."""
    return math.exp(x)

def derivative_exp(x: Number) -> float:
    """Returns the derivative of the exponential of x."""
    return math.exp(x)

def log(x: Number) -> float:
    """Returns the natural logarithm of x."""
    if x <= 0:
        raise ValueError("Logarithm is only defined for positive numbers")
    return math.log(x)

def derivative_log(x: Number) -> float:
    """Returns the derivative of the natural logarithm of x."""
    if x <= 0:
        raise ValueError("Derivative of log is only defined for positive numbers")
    return 1 / x



# list_utils.py

"""
A collection of utility functions for manipulating lists and other sequences.
"""
from typing import Any, List, TypeVar, Union

T = TypeVar("T")
Number = Union[int, float]

def chunk_list(items: List[T], n: int) -> List[List[T]]:
    """
    Breaks a list into chunks of size n.

    :param items: The list to be chunked.
    :param n: The desired size of each chunk.
    :return: A list of lists, where each inner list is a chunk.
    
    >>> chunk_list([1, 2, 3, 4, 5, 6, 7, 8, 9], 4)
    [[1, 2, 3, 4], [5, 6, 7, 8], [9]]
    """
    if n <= 0:
        raise ValueError("Chunk size n must be a positive integer.")
    return [items[i:i + n] for i in range(0, len(items), n)]


def remove_empty_lists(items: List[Any]) -> List[Any]:
    """
    Removes all empty lists from a list of items.
    
    >>> remove_empty_lists([5, 6, [], 3, [], [], 9])
    [5, 6, 3, 9]
    """
    return [item for item in items if item != []]


def get_top_element(items: List[Number]) -> Number:
    """
    Returns the largest number in a list.
    
    Note: This was named biggest_no, and the implementation was flawed.
    Using the built-in max() is the correct and most efficient approach.
    """
    if not items:
        raise ValueError("Cannot get top element from an empty list.")
    return max(items)


def merge_lists_pairwise(list1: List[T], list2: List[Any]) -> List[tuple[T, Any]]:
    """Merges two lists into a list of pairs (tuples)."""
    return list(zip(list1, list2))


def append_lists(list1: List[T], list2: List[T]) -> List[T]:
    """
    Creates a new list by appending the second list to the first.
    
    Note: The original implementation used .extend(), which modifies the list
    in-place and returns None. This version returns a new concatenated list.
    """
    return list1 + list2


def reverse_list(items: List[T]) -> List[T]:
    """Returns a reversed copy of the list."""
    return items[::-1]


def add_list_elements(list1: List[Number], list2: List[Number]) -> List[Number]:
    """Adds two lists element-wise."""
    return [i + j for i, j in zip(list1, list2)]


def subtract_list_elements(list1: List[Number], list2: List[Number]) -> List[Number]:
    """Subtracts the second list from the first, element-wise."""
    return [i - j for i, j in zip(list1, list2)]


def add_if_even(list1: List[int], list2: List[int]) -> List[int]:
    """Adds list elements if both corresponding elements are even."""
    return [i + j for i, j in zip(list1, list2) if i % 2 == 0 and j % 2 == 0]


def multiply_if_odd(list1: List[int], list2: List[int]) -> List[int]:
    """Multiplies list elements if both corresponding elements are odd."""
    return [i * j for i, j in zip(list1, list2) if i % 2 != 0 and j % 2 != 0]


def list_element_power(items: List[Number], power: int) -> List[Number]:
    """Raises each element in a list to the given power."""
    return [item**power for item in items]



# dict_utils.py

"""
A collection of utility functions for manipulating dictionaries.
"""
from typing import Any, Dict, List, Tuple, TypeVar, Hashable

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


def is_value_present(data: Dict[Any, V], value: V) -> bool:
    """Checks if a value is present in any of the dictionary's values."""
    return value in data.values()


def count_value_occurrences(data: Dict[Any, V], value: V) -> int:
    """Counts how many times a value appears in the dictionary's values."""
    return list(data.values()).count(value)


def sort_dict_by_value(data: Dict[K, V], descending: bool = False) -> Dict[K, V]:
    """
    Sorts a dictionary by its values.

    :param data: The dictionary to sort.
    :param descending: If True, sort in descending order.
    :return: A new dictionary sorted by value.
    """
    sorted_items = sorted(data.items(), key=lambda item: item[1], reverse=descending)
    return dict(sorted_items)


def sort_dict_by_key(data: Dict[K, V], descending: bool = False) -> Dict[K, V]:
    """
    Sorts a dictionary by its keys.

    :param data: The dictionary to sort.
    :param descending: If True, sort in descending order.
    :return: A new dictionary sorted by key.
    """
    sorted_items = sorted(data.items(), key=lambda item: item[0], reverse=descending)
    return dict(sorted_items)
    
def merge_dicts(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merges two dictionaries. Keys from dict2 will overwrite those in dict1.
    
    Note: The original implementation used .update(), which modifies a dict
    in-place and returns None. This version returns a new merged dict.
    """
    merged = dict1.copy()
    merged.update(dict2)
    return merged

def list_of_tuples_to_dict(tuples: List[Tuple[K, V]]) -> Dict[K, List[V]]:
    """
    Converts a list of key-value tuples into a dictionary where keys
    can have multiple values.
    
    >>> list_of_tuples_to_dict([("A", 10), ("B", 20), ("A", 30)])
    {'A': [10, 30], 'B': [20]}
    """
    result: Dict[K, List[V]] = {}
    for key, value in tuples:
        result.setdefault(key, []).append(value)
    return result



# string_utils.py

"""
A collection of utility functions for string manipulation and processing.
"""
import re
from typing import List, Set

def split_to_chars(word: str) -> List[str]:
    """Splits a word into a list of its characters."""
    return list(word)


def strip_vowels(text: str) -> str:
    """Removes all vowels (case-insensitive) from a string."""
    vowels = "aeiouAEIOU"
    return "".join(char for char in text if char not in vowels)


def clean_string(text: str) -> str:
    """Removes all non-alphanumeric characters from a string."""
    return re.sub(r'[\W_]+', '', text)


def replace_substring(text: str, old: str, new: str) -> str:
    """
    Replaces all occurrences of a substring with a new one.
    
    Note: The original implementation was missing the `return` statement.
    """
    return text.replace(old, new)


def get_word_frequencies(text: str) -> dict[str, int]:
    """
    Calculates the frequency of each word in a string.

    :param text: The input string.
    :return: A dictionary mapping each word to its count.
    
    >>> get_word_frequencies('times of india times new india express')
    {'times': 2, 'of': 1, 'india': 2, 'new': 1, 'express': 1}
    """
    words = text.split()
    return {word: words.count(word) for word in set(words)}


def contains_all_vowels(text: str) -> bool:
    """Checks if a string contains all vowels (a, e, i, o, u), case-insensitive."""
    return set(text.lower()).issuperset(set("aeiou"))


def rotate_string(text: str, d: int) -> Tuple[str, str]:
    """
    Rotates a string left and right by d characters.
    
    :param text: The string to rotate.
    :param d: The number of positions to rotate by.
    :return: A tuple containing the (left_rotated_string, right_rotated_string).
    """
    d = d % len(text)
    left_rotated = text[d:] + text[:d]
    right_rotated = text[-d:] + text[:-d]
    return left_rotated, right_rotated

def remove_duplicate_words(text: str) -> str:
    """
    Removes duplicate words from a sentence, preserving order of first appearance.
    
    >>> remove_duplicate_words("Hello world Hello again world")
    'Hello world again'
    """
    words = text.split()
    ordered_unique_words = list(dict.fromkeys(words))
    return ' '.join(ordered_unique_words)


def find_even_length_words(text: str) -> List[str]:
    """Finds all words with an even length in a string."""
    return [word for word in text.split() if len(word) % 2 == 0]


def remove_duplicate_chars(text: str, preserve_order: bool = True) -> str:
    """
    Removes duplicate characters from a string.
    
    :param text: The input string.
    :param preserve_order: If True, keeps the first occurrence of each char.
                           If False, the order is not guaranteed.
    :return: A string with unique characters.
    """
    if preserve_order:
        return "".join(dict.fromkeys(text))
    else:
        return "".join(set(text))
        
def get_max_frequency_char(text: str) -> str:
    """
    Finds the character with the highest frequency in a string.
    If there's a tie, one of the tied characters is returned.
    """
    if not text:
        raise ValueError("Input string cannot be empty.")
    return max(set(text), key=text.count)

def has_special_chars(text: str) -> bool:
    """Checks if a string contains any special characters."""
    regex = re.compile('[@_!#$%^&*()<>?/\|}{~:]')
    return regex.search(text) is not None

def is_binary_string(text: str) -> bool:
    """Checks if a string contains only '0's and '1's."""
    return set(text).issubset({'0', '1'})

def is_palindrome(text: str) -> bool:
    """
    Checks if a string is a palindrome (reads the same forwards and backwards).
    This check is case-sensitive.
    """
    return text == text[::-1]



# data_utils.py

"""
Utilities for data input/output operations, such as handling
CSV files and pickling data.
"""
import csv
import pickle
from typing import Any, List


def read_csv_file(filename: str) -> List[List[str]]:
    """
    Reads a CSV file and returns its content as a list of rows.

    :param filename: The path to the CSV file.
    :return: A list where each item is a list of strings representing a row.
    """
    with open(filename, mode='r', newline='', encoding='utf-8') as f:
        reader = csv.reader(f)
        return list(reader)


def pickle_data(data: Any, pickle_file: str) -> None:
    """
    Serializes data and saves it to a file using pickle.

    :param data: The Python object to pickle.
    :param pickle_file: The path to the output file.
    """
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle_data(pickle_file: str) -> Any:
    """
    Loads and deserializes data from a pickle file.

    :param pickle_file: The path to the pickle file.
    :return: The deserialized Python object.
    """
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)



# utils.py

"""
A collection of miscellaneous utility functions, including timing,
memory management, and date/time operations.
"""
import gc
import time
from datetime import datetime, timedelta
from typing import Any, Callable


def clear_memory() -> None:
    """Runs the garbage collector to free up memory."""
    gc.collect()


def time_it(func: Callable, *args: Any, repetitions: int = 1, **kwargs: Any) -> float:
    """
    Measures the average execution time of a function.

    :param func: The function to be timed.
    :param args: Positional arguments to pass to the function.
    :param repetitions: The number of times to run the function for averaging.
    :param kwargs: Keyword arguments to pass to the function.
    :return: The average execution time in seconds.
    """
    if repetitions <= 0:
        raise ValueError("Repetitions must be a positive integer.")
    
    total_time = 0.0
    for _ in range(repetitions):
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time += (end_time - start_time)
        
    return total_time / repetitions

def get_relative_dates() -> dict[str, str]:
    """
    Gets yesterday's, today's, and tomorrow's date.
    
    :return: A dictionary with 'yesterday', 'today', and 'tomorrow' dates
             formatted as 'dd-mm-YYYY'.
    """
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    tomorrow = today + timedelta(days=1)
    
    date_format = '%d-%m-%Y'
    return {
        "yesterday": yesterday.strftime(date_format),
        "today": today.strftime(date_format),
        "tomorrow": tomorrow.strftime(date_format),
    }

def convert_12h_to_24h(time_str: str) -> str:
    """
    Converts a time string from 12-hour format (e.g., "08:05:45 PM")
    to 24-hour format (e.g., "20:05:45").
    """
    dt_object = datetime.strptime(time_str, "%I:%M:%S %p")
    return dt_object.strftime("%H:%M:%S")



# main.py

"""
Demonstration script for the Python utility functions.

This script imports functions from the various utility modules and shows
how to use them with example data.
"""

# Import all the refactored functions
from itertools_recipes import pairwise, grouper, powerset
from list_utils import chunk_list, get_top_element, reverse_list
from dict_utils import sort_dict_by_value, is_value_present, get_word_frequencies
from string_utils import strip_vowels, is_palindrome, rotate_string
from math_helpers import grade, factorial
from utils import get_relative_dates, convert_12h_to_24h

def main():
    """Main function to run all demonstrations."""
    print("--- Demonstrating Utility Functions ---\n")

    # --- itertools_recipes ---
    print("--- itertools_recipes.py ---")
    print(f"pairwise('ABCDE'): {list(pairwise('ABCDE'))}")
    print(f"grouper('ABCDEFG', 3, 'x'): {list(grouper('ABCDEFG', 3, 'x'))}")
    print(f"powerset([1, 2, 3]): {list(powerset([1, 2, 3]))}")
    print("-" * 20)

    # --- list_utils ---
    print("\n--- list_utils.py ---")
    my_list = [10, 20, 4, 45, 99]
    print(f"Original list: {my_list}")
    print(f"Reversed list: {reverse_list(my_list)}")
    print(f"Largest element: {get_top_element(my_list)}")
    print(f"List chunked into size 3: {chunk_list(my_list, 3)}")
    print("-" * 20)

    # --- dict_utils ---
    print("\n--- dict_utils.py ---")
    my_dict = {'a': 100, 'c': 50, 'b': 200}
    print(f"Original dictionary: {my_dict}")
    print(f"Is value 50 present? {is_value_present(my_dict, 50)}")
    print(f"Sorted by value: {sort_dict_by_value(my_dict)}")
    sentence = 'times of india times new india express'
    print(f"Word frequencies for '{sentence}': {get_word_frequencies(sentence)}")
    print("-" * 20)

    # --- string_utils ---
    print("\n--- string_utils.py ---")
    palindrome_str = "malayalam"
    regular_str = "hello"
    print(f"Is '{palindrome_str}' a palindrome? {is_palindrome(palindrome_str)}")
    print(f"Is '{regular_str}' a palindrome? {is_palindrome(regular_str)}")
    print(f"'Programming' with vowels stripped: {strip_vowels('Programming')}")
    left, right = rotate_string('helloworld', 2)
    print(f"Rotating 'helloworld' by 2: Left='{left}', Right='{right}'")
    print("-" * 20)

    # --- math_helpers ---
    print("\n--- math_helpers.py ---")
    print(f"Grade for a score of 88: '{grade(88)}'")
    print(f"Factorial of 5: {factorial(5)}")
    print("-" * 20)
    
    # --- utils ---
    print("\n--- utils.py ---")
    print(f"Relative dates: {get_relative_dates()}")
    time_12h = "08:05:45 PM"
    print(f"Time '{time_12h}' in 24h format: {convert_12h_to_24h(time_12h)}")
    print("-" * 20)


if __name__ == "__main__":
    main()
