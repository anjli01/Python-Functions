# python_recipes.py
"""
A collection of Python functions and snippets demonstrating common patterns and recipes.

This module is organized into several sections:
1.  Itertools & Functional Programming Recipes
2.  File I/O Utilities
3.  String Manipulation Functions
4.  Dictionary Utilities
5.  List & Collection Utilities
6.  Miscellaneous Utilities
7.  Demonstration of usage in a main execution block.
"""

# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import bisect
import collections
import csv
import datetime
import itertools
import math
import operator
import pickle
import random
import re
import time
from typing import (Any, Callable, Dict, Iterable, Iterator, List,
                    Optional, Set, Tuple, TypeVar, Union)

# Define generic types for better type hinting
T = TypeVar('T')
K = TypeVar('K')
V = TypeVar('V')


# ==============================================================================
# 2. ITERTOOLS & FUNCTIONAL PROGRAMMING RECIPES
# Inspired by the Python official documentation for the itertools module.
# ==============================================================================

def n_cycles(iterable: Iterable[T], n: int) -> Iterator[T]:
    """
    Returns the sequence elements n times.

    Args:
        iterable: The iterable to repeat.
        n: The number of times to repeat the iterable.

    Returns:
        An iterator that yields elements from the iterable n times.

    Example:
        >>> list(n_cycles([1, 2], 3))
        [1, 2, 1, 2, 1, 2]
    """
    return itertools.chain.from_iterable(itertools.repeat(tuple(iterable), n))


def dot_product(vec1: Iterable[float], vec2: Iterable[float]) -> float:
    """
    Calculates the dot product of two vectors.

    Args:
        vec1: The first vector (an iterable of numbers).
        vec2: The second vector (an iterable of numbers).

    Returns:
        The dot product of the two vectors.

    Example:
        >>> dot_product([1, 2, 3], [4, 5, 6])
        32
    """
    return sum(map(operator.mul, vec1, vec2))


def flatten(list_of_lists: Iterable[Iterable[T]]) -> Iterator[T]:
    """
    Flattens one level of nesting in an iterable of iterables.

    Args:
        list_of_lists: An iterable where each element is itself an iterable.

    Returns:
        An iterator that yields elements from the nested iterables.

    Example:
        >>> list(flatten([[1, 2], [3, 4], [5]]))
        [1, 2, 3, 4, 5]
    """
    return itertools.chain.from_iterable(list_of_lists)


def repeat_func(func: Callable, times: Optional[int] = None, *args: Any) -> Iterator[Any]:
    """
    Repeats calls to a function with specified arguments.

    Args:
        func: The function to call repeatedly.
        times: The number of times to call the function. If None, it repeats indefinitely.
        *args: The arguments to pass to the function.

    Returns:
        An iterator that yields the results of calling func.
    """
    if times is None:
        return itertools.starmap(func, itertools.repeat(args))
    return itertools.starmap(func, itertools.repeat(args, times))


def pairwise(iterable: Iterable[T]) -> Iterator[Tuple[T, T]]:
    """
    Returns successive overlapping pairs from an iterable.

    Args:
        iterable: The source iterable.

    Returns:
        An iterator of tuples, where each tuple contains two adjacent elements.

    Example:
        >>> list(pairwise('ABCDE'))
        [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')]
    """
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)


def grouper(iterable: Iterable[T], n: int, fillvalue: Optional[T] = None) -> Iterator[Tuple[T, ...]]:
    """
    Collects data into fixed-length chunks or blocks.

    Args:
        iterable: The source iterable.
        n: The size of each chunk.
        fillvalue: The value to use for padding the last chunk if it's short.

    Returns:
        An iterator of tuples, where each tuple is a chunk of size n.

    Example:
        >>> list(grouper('ABCDEFG', 3, 'x'))
        [('A', 'B', 'C'), ('D', 'E', 'F'), ('G', 'x', 'x')]
    """
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def roundrobin(*iterables: Iterable[T]) -> Iterator[T]:
    """
    Yields elements from multiple iterables in a round-robin fashion.

    Args:
        *iterables: One or more iterables.

    Returns:
        An iterator that yields elements from the input iterables one by one.

    Example:
        >>> list(roundrobin('ABC', 'D', 'EF'))
        ['A', 'D', 'E', 'B', 'F', 'C']
    """
    # Recipe credited to George Sakkis
    num_active = len(iterables)
    nexts = itertools.cycle(iter(it).__next__ for it in iterables)
    while num_active:
        try:
            for next_item in nexts:
                yield next_item()
        except StopIteration:
            num_active -= 1
            nexts = itertools.cycle(itertools.islice(nexts, num_active))


def partition(pred: Callable[[T], bool], iterable: Iterable[T]) -> Tuple[Iterator[T], Iterator[T]]:
    """
    Partitions entries of an iterable based on a predicate.

    Args:
        pred: A function that returns True or False for an element.
        iterable: The iterable to partition.

    Returns:
        A tuple containing two iterators: one for elements where pred is False,
        and one for elements where pred is True.

    Example:
        >>> is_odd = lambda x: x % 2 != 0
        >>> false_items, true_items = partition(is_odd, range(10))
        >>> list(false_items)
        [0, 2, 4, 6, 8]
        >>> list(true_items)
        [1, 3, 5, 7, 9]
    """
    t1, t2 = itertools.tee(iterable)
    return itertools.filterfalse(pred, t1), filter(pred, t2)


def powerset(iterable: Iterable[T]) -> Iterator[Tuple[T, ...]]:
    """
    Generates the powerset of an iterable.

    Args:
        iterable: The input iterable.

    Returns:
        An iterator yielding all subsets of the iterable as tuples.

    Example:
        >>> list(powerset([1, 2, 3]))
        [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    s = list(iterable)
    return itertools.chain.from_iterable(itertools.combinations(s, r) for r in range(len(s) + 1))


def unique_everseen(iterable: Iterable[T], key: Optional[Callable[[T], Any]] = None) -> Iterator[T]:
    """
    Yields unique elements, preserving order. Remembers all elements seen.

    Args:
        iterable: The source iterable.
        key: A function to extract a comparison key from elements.

    Returns:
        An iterator of unique elements.

    Example:
        >>> list(unique_everseen('AAAABBBCCDAABBB'))
        ['A', 'B', 'C', 'D']
        >>> list(unique_everseen('ABBCcAD', str.lower))
        ['A', 'B', 'C', 'D']
    """
    seen: Set[Any] = set()
    seen_add = seen.add
    if key is None:
        for element in itertools.filterfalse(seen.__contains__, iterable):
            seen_add(element)
            yield element
    else:
        for element in iterable:
            k = key(element)
            if k not in seen:
                seen_add(k)
                yield element


def unique_justseen(iterable: Iterable[T], key: Optional[Callable[[T], Any]] = None) -> Iterator[T]:
    """
    Yields unique elements, preserving order. Remembers only the last element seen.

    Args:
        iterable: The source iterable.
        key: A function to extract a comparison key from elements.

    Returns:
        An iterator of unique elements based on consecutive grouping.

    Example:
        >>> list(unique_justseen('AAAABBBCCDAABBB'))
        ['A', 'B', 'C', 'D', 'A', 'B']
        >>> list(unique_justseen('ABBCcAD', str.lower))
        ['A', 'B', 'C', 'A', 'D']
    """
    return (next(g) for k, g in itertools.groupby(iterable, key))


def iter_except(func: Callable[[], T], exception: type, first: Optional[Callable[[], T]] = None) -> Iterator[T]:
    """
    Converts a call-until-exception interface to an iterator interface.

    Like `iter(func, sentinel)` but uses an exception to end the loop.

    Args:
        func: The function to call.
        exception: The exception type that signals the end of iteration.
        first: An optional function to call to get the first item.

    Yields:
        Items from `func` until `exception` is raised.
    """
    try:
        if first:
            yield first()
        while True:
            yield func()
    except exception:
        pass


def random_product(*args: Iterable[T], repeat: int = 1) -> Tuple[T, ...]:
    """
    Returns a random selection from itertools.product(*args, **kwds).

    Args:
        *args: The iterables to choose from.
        repeat: The number of times to repeat the iterables.

    Returns:
        A tuple representing one random element from the Cartesian product.
    """
    pools = [tuple(pool) for pool in args] * repeat
    return tuple(random.choice(pool) for pool in pools)


def random_permutation(iterable: Iterable[T], r: Optional[int] = None) -> Tuple[T, ...]:
    """
    Returns a random r-length permutation from an iterable.

    Args:
        iterable: The source iterable.
        r: The length of the permutation. Defaults to the length of the iterable.

    Returns:
        A random r-length tuple.
    """
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def random_combination(iterable: Iterable[T], r: int) -> Tuple[T, ...]:
    """
    Returns a random r-length combination from an iterable.

    Args:
        iterable: The source iterable.
        r: The length of the combination.

    Returns:
        A random r-length tuple.
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        raise ValueError("r must be less than or equal to the length of the iterable.")
    indices = sorted(random.sample(range(n), r))
    return tuple(pool[i] for i in indices)


def random_combination_with_replacement(iterable: Iterable[T], r: int) -> Tuple[T, ...]:
    """
    Returns a random r-length combination with replacement.

    Args:
        iterable: The source iterable.
        r: The length of the combination.

    Returns:
        A random r-length tuple with replacement.
    """
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        raise ValueError("Cannot choose from an empty sequence.")
    indices = sorted(random.choices(range(n), k=r))
    return tuple(pool[i] for i in indices)


# ==============================================================================
# 3. BISECT MODULE RECIPES
# Functions for searching in sorted lists.
# ==============================================================================

def index(a: List[T], x: T) -> int:
    """Locates the leftmost value exactly equal to x in a sorted list."""
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise ValueError(f"{x} is not in list")


def find_lt(a: List[T], x: T) -> T:
    """Finds the rightmost value less than x in a sorted list."""
    i = bisect.bisect_left(a, x)
    if i:
        return a[i - 1]
    raise ValueError(f"No value less than {x} in list")


def find_le(a: List[T], x: T) -> T:
    """Finds the rightmost value less than or equal to x in a sorted list."""
    i = bisect.bisect_right(a, x)
    if i:
        return a[i - 1]
    raise ValueError(f"No value less than or equal to {x} in list")


def find_gt(a: List[T], x: T) -> T:
    """Finds the leftmost value greater than x in a sorted list."""
    i = bisect.bisect_right(a, x)
    if i != len(a):
        return a[i]
    raise ValueError(f"No value greater than {x} in list")


def find_ge(a: List[T], x: T) -> T:
    """Finds the leftmost item greater than or equal to x in a sorted list."""
    i = bisect.bisect_left(a, x)
    if i != len(a):
        return a[i]
    raise ValueError(f"No value greater than or equal to {x} in list")


def assign_grade(score: int, breakpoints: List[int] = [60, 70, 80, 90], grades: str = 'FDCBA') -> str:
    """Maps a numeric score to a letter grade using bisect."""
    i = bisect.bisect(breakpoints, score)
    return grades[i]


# ==============================================================================
# 4. FILE I/O UTILITIES
# ==============================================================================

def read_csv_file(filename: str) -> None:
    """Reads a CSV file and prints its content row by row."""
    try:
        with open(filename, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                print(row)
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")


def pickle_data(data: Any, pickle_file: str) -> None:
    """Serializes data to a file using pickle."""
    with open(pickle_file, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)


def load_pickle_data(pickle_file: str) -> Any:
    """Deserializes data from a pickle file."""
    with open(pickle_file, 'rb') as f:
        return pickle.load(f)


# ==============================================================================
# 5. STRING MANIPULATION FUNCTIONS
# ==============================================================================

def is_palindrome(s: str) -> bool:
    """Checks if a given string is a palindrome."""
    return s == s[::-1]


def is_symmetrical(s: str) -> bool:
    """Checks if a string is symmetrical (first half == second half)."""
    n = len(s)
    half = n // 2
    first_half = s[:half]
    second_half = s[n - half:]
    return first_half == second_half


def reverse_words_in_sentence(sentence: str) -> str:
    """Reverses the order of words in a sentence string."""
    words = sentence.split()
    return ' '.join(reversed(words))


def get_even_length_words(text: str) -> List[str]:
    """Returns a list of words with even length from a string."""
    return [word for word in text.split() if len(word) % 2 == 0]


def contains_all_vowels(text: str, vowels: str = "aeiou") -> bool:
    """Checks if a string contains all vowels (case-insensitive)."""
    return set(vowels).issubset(set(text.lower()))


def count_unique_matching_chars(str1: str, str2: str) -> int:
    """Returns the count of unique characters common to both strings."""
    return len(set(str1) & set(str2))


def remove_duplicate_chars(text: str) -> str:
    """Removes duplicate characters from a string, preserving order."""
    return "".join(dict.fromkeys(text))


def get_char_frequencies(text: str) -> Dict[str, int]:
    """Calculates the frequency of each character in a string."""
    return collections.Counter(text)


def get_word_frequencies(text: str) -> Dict[str, int]:
    """Calculates the frequency of each word in a string."""
    return collections.Counter(text.split())


def find_least_frequent_char(text: str) -> Optional[str]:
    """Finds the least frequent character in a string."""
    if not text:
        return None
    counts = get_char_frequencies(text)
    return min(counts, key=counts.get)


def find_most_frequent_char(text: str) -> Optional[str]:
    """Finds the most frequent character in a string."""
    if not text:
        return None
    counts = get_char_frequencies(text)
    return max(counts, key=counts.get)


def find_words_shorter_than(text: str, max_len: int) -> List[str]:
    """Finds all words in a string shorter than a given length."""
    return [word for word in text.split() if len(word) < max_len]


def is_binary_string(text: str) -> bool:
    """Checks if a string contains only '0's and '1's."""
    return all(char in '01' for char in text)


def remove_char_at_index(text: str, i: int) -> str:
    """Removes the character at a specific index in a string."""
    if 0 <= i < len(text):
        return text[:i] + text[i + 1:]
    return text  # Or raise IndexError


def find_urls(text: str) -> List[str]:
    """Finds all URLs in a given string."""
    # This regex is a common and reasonably effective one.
    url_pattern = re.compile(
        r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)"
        r"(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+"
        r"(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    )
    return [match[0] for match in url_pattern.findall(text)]


def find_uncommon_words(str1: str, str2: str) -> List[str]:
    """Finds words that appear in exactly one of the two strings."""
    counts = collections.Counter(str1.split())
    counts.update(str2.split())
    return [word for word, count in counts.items() if count == 1]


def rotate_string(text: str, n: int) -> str:
    """Rotates a string left or right by n positions."""
    if not text:
        return ""
    n = n % len(text)  # Normalize n
    return text[n:] + text[:n]


def move_digits_to_end(text: str) -> str:
    """Moves all numeric digits to the end of the string."""
    letters = ''.join(filter(str.isalpha, text))
    digits = ''.join(filter(str.isdigit, text))
    others = ''.join(filter(lambda c: not c.isalnum(), text))
    return letters + others + digits


def increment_trailing_number(text: str) -> str:
    """Increments a number at the end of a string, preserving padding."""
    def replacer(match: re.Match) -> str:
        number_str = match.group()
        new_number = int(number_str) + 1
        return str(new_number).zfill(len(number_str))

    return re.sub(r'\d+$', replacer, text)


def count_letters_and_digits(text: str) -> Tuple[int, int]:
    """Calculates the number of letters and digits in a string."""
    letters = sum(c.isalpha() for c in text)
    digits = sum(c.isdigit() for c in text)
    return letters, digits


def has_lowercase(text: str) -> bool:
    """Checks if a string contains at least one lowercase letter."""
    return any(c.islower() for c in text)


def has_uppercase(text: str) -> bool:
    """Checks if a string contains at least one uppercase letter."""
    return any(c.isupper() for c in text)


# ==============================================================================
# 6. DICTIONARY UTILITIES
# ==============================================================================

def lists_to_dict(keys: List[K], values: List[V]) -> Dict[K, V]:
    """Converts two lists of equal length into a dictionary."""
    return dict(zip(keys, values))


def sort_list_of_dicts(dict_list: List[Dict], sort_key: str) -> None:
    """Sorts a list of dictionaries in-place by a specified key."""
    dict_list.sort(key=lambda item: item.get(sort_key))


def get_longest_key(data: Dict) -> Optional[Any]:
    """Returns the longest key in a dictionary."""
    if not data:
        return None
    return max(data, key=len)


def capitalize_key_ends(data: Dict[str, V]) -> Dict[str, V]:
    """Capitalizes the first and last character of each string key in a dictionary."""
    return {
        (key[0].upper() + key[1:-1] + key[-1].upper() if len(key) > 1 else key.upper()): value
        for key, value in data.items()
    }


def invert_dict(data: Dict[K, V]) -> Dict[V, K]:
    """
    Inverts a dictionary. Assumes values are unique and hashable.
    """
    return {value: key for key, value in data.items()}


def invert_dict_with_list(data: Dict[K, V]) -> Dict[V, List[K]]:
    """
    Inverts a dictionary with non-unique values.
    Keys mapping to the same value are appended to a list.
    """
    inverted: Dict[V, List[K]] = {}
    for key, value in data.items():
        inverted.setdefault(value, []).append(key)
    return inverted


def merge_list_of_dicts(dict_list: List[Dict[K, V]]) -> Dict[K, V]:
    """Merges a list of dictionaries into a single dictionary."""
    return {key: value for d in dict_list for key, value in d.items()}


def mean_key_value_length_diff(data: Dict[str, str]) -> float:
    """
    Returns the mean difference in the length of keys and values.
    Assumes a dictionary of strings.
    """
    if not data:
        return 0.0
    sum_diff = sum(abs(len(val) - len(key)) for key, val in data.items())
    return sum_diff / len(data)


def count_unique_keys(dict_list: List[Dict]) -> int:
    """Counts the number of unique keys in a list of dictionaries."""
    return len({key for d in dict_list for key in d})


def flatten_dict(nested_dict: Dict, separator: str = '_', prefix: str = '') -> Dict:
    """Flattens a nested dictionary by joining keys with a separator."""
    items = []
    for k, v in nested_dict.items():
        new_key = prefix + separator + k if prefix else k
        if isinstance(v, collections.abc.Mapping):
            items.extend(flatten_dict(v, separator, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def combine_dicts_with_sum(d1: Dict[K, int], d2: Dict[K, int]) -> Dict[K, int]:
    """
    Combines two dictionaries by adding values for common keys.
    """
    combined = d1.copy()
    for key, value in d2.items():
        combined[key] = combined.get(key, 0) + value
    return combined


# ==============================================================================
# 7. LIST & COLLECTION UTILITIES
# ==============================================================================

def add_lists_elementwise(list1: List[float], list2: List[float]) -> List[float]:
    """Adds elements of two lists of the same length."""
    return [x + y for x, y in zip(list1, list2)]


def multiply_list_elements(numbers: List[Union[int, float]]) -> Union[int, float]:
    """Multiplies all the numbers in a list."""
    # math.prod is available in Python 3.8+
    return math.prod(numbers)


def get_positive_numbers(numbers: List[Union[int, float]]) -> List[Union[int, float]]:
    """Filters a list to return only the positive numbers."""
    return [num for num in numbers if num >= 0]


def product_of_integers_in_mixed_list(mixed_list: List[Any]) -> int:
    """Calculates the product of all integers in a list of mixed types."""
    product = 1
    for item in mixed_list:
        if isinstance(item, int):
            product *= item
    return product


# ==============================================================================
# 8. MAIN EXECUTION BLOCK (DEMONSTRATIONS)
# ==============================================================================

if __name__ == "__main__":
    print("--- Running Demonstrations for Python Recipes ---\n")

    # --- Itertools & Functional Programming Demos ---
    print("--- Itertools & Functional Demos ---")
    print(f"n_cycles([1, 2], 3): {list(n_cycles([1, 2], 3))}")
    print(f"dot_product([1, 2, 3], [4, 5, 6]): {dot_product([1, 2, 3], [4, 5, 6])}")
    print(f"flatten([[1, 2], [3, 4]]): {list(flatten([[1, 2], [3, 4]]))}")
    print(f"pairwise('ABCDE'): {list(pairwise('ABCDE'))}")
    print(f"roundrobin('ABC', 'D', 'EF'): {list(roundrobin('ABC', 'D', 'EF'))}")
    is_odd = lambda x: x % 2 != 0
    false_items, true_items = partition(is_odd, range(10))
    print(f"partition(is_odd, range(10)): {list(false_items)}, {list(true_items)}")
    print(f"powerset([1, 2]): {list(powerset([1, 2]))}")
    print(f"unique_everseen('ABBCcAD', str.lower): {list(unique_everseen('ABBCcAD', str.lower))}")
    print(f"unique_justseen('AAAABBBCCDAABBB'): {list(unique_justseen('AAAABBBCCDAABBB'))}")
    print("-" * 20)

    # --- String Manipulation Demos ---
    print("\n--- String Manipulation Demos ---")
    print(f"is_palindrome('madam'): {is_palindrome('madam')}")
    print(f"is_symmetrical('abccba'): {is_symmetrical('abccba')}")
    sentence = "python is a great language"
    print(f"reverse_words_in_sentence('{sentence}'): '{reverse_words_in_sentence(sentence)}'")
    print(f"find_least_frequent_char('hello world'): '{find_least_frequent_char('hello world')}'")
    print(f"remove_duplicate_chars('programming'): '{remove_duplicate_chars('programming')}'")
    print(f"is_binary_string('010110'): {is_binary_string('010110')}")
    print(f"increment_trailing_number('file_009'): '{increment_trailing_number('file_009')}'")
    letters, digits = count_letters_and_digits('python1234')
    print(f"count_letters_and_digits('python1234'): Letters={letters}, Digits={digits}")
    print("-" * 20)
    
    # --- Dictionary Utilities Demos ---
    print("\n--- Dictionary Utilities Demos ---")
    d1 = {'a': 1, 'b': 2}
    d2 = {'b': 3, 'c': 4}
    print(f"combine_dicts_with_sum({d1}, {d2}): {combine_dicts_with_sum(d1, d2)}")
    nested = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    print(f"flatten_dict({nested}): {flatten_dict(nested)}")
    non_unique_dict = {'a': 1, 'b': 2, 'c': 1}
    print(f"invert_dict_with_list({non_unique_dict}): {invert_dict_with_list(non_unique_dict)}")
    print("-" * 20)

    # --- List & Collection Demos ---
    print("\n--- List & Collection Demos ---")
    mixed = [5, 8, "gfg", 8, (5, 7), 'is', 2, 3]
    print(f"product_of_integers_in_mixed_list({mixed}): {product_of_integers_in_mixed_list(mixed)}")
    print(f"multiply_list_elements([1, 2, 3, 4]): {multiply_list_elements([1, 2, 3, 4])}")
    nums = [11, -21, 0, 45, 66, -93]
    print(f"get_positive_numbers({nums}): {get_positive_numbers(nums)}")
    print("-" * 20)
