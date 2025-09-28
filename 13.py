# Refactored Code with Standard Practices

import math
import collections
import functools
import heapq
import itertools
import string
from typing import (Any, Callable, Dict, Generator, List, Optional, Set,
                    Tuple, Union)

# ==============================================================================
# Section: Basic Functions & Utilities
# ==============================================================================


def get_ascii_value(character: str) -> int:
    """
    Returns the ASCII value of the passed-in character.

    Args:
        character: A single character string.

    Returns:
        The integer ASCII value.
    """
    if len(character) != 1:
        raise ValueError("Input must be a single character.")
    return ord(character)


def union_of_sets(set1: Set[Any], set2: Set[Any]) -> Set[Any]:
    """
    Creates a union of two sets.

    Args:
        set1: The first set.
        set2: The second set.

    Returns:
        A new set containing all elements from both sets.
    """
    return set1.union(set2)  # .union() is often more readable than |


def add_lists_elementwise(list1: List[float], list2: List[float]) -> List[float]:
    """
    Adds two lists of numbers element-wise.

    Note: The lists must be of the same length.

    Args:
        list1: The first list of numbers.
        list2: The second list of numbers.

    Returns:
        A new list where each element is the sum of the elements
        at the same index from the input lists.
    """
    if len(list1) != len(list2):
        raise ValueError("Lists must be of the same length.")
    return [x + y for x, y in zip(list1, list2)]


def _is_divisible_by_all_digits(n: int) -> bool:
    """Helper function to check if a number is divisible by each of its digits."""
    if n == 0:
        return True
    s_n = str(n)
    if '0' in s_n:
        return False
    return all(n % int(digit) == 0 for digit in s_n)


def find_numbers_divisible_by_digits(start: int, end: int) -> List[int]:
    """
    Finds numbers within a given range where each number is divisible by
    every digit it contains.

    Args:
        start: The starting number of the range (inclusive).
        end: The ending number of the range (inclusive).

    Returns:
        A list of numbers that satisfy the condition.
    """
    return [
        n for n in range(start, end + 1) if _is_divisible_by_all_digits(n)
    ]


def find_max_in_heterogeneous_list(items: List[Any]) -> Any:
    """
    Finds the maximum value in a heterogeneous list. Integers are
    prioritized over other types.

    Args:
        items: A list containing values of different types.

    Returns:
        The maximum value found in the list.
    """
    if not items:
        return None
    # The key sorts by two criteria:
    # 1. Is the item an int? (True > False, so ints come first)
    # 2. The value of the item itself.
    return max(items, key=lambda i: (isinstance(i, (int, float)), i))


def get_first_and_last_char(text: str) -> str:
    """
    Takes a string and returns the concatenated first and last characters.

    Args:
        text: The input string.

    Returns:
        A string made of the first and last characters.
    """
    if len(text) < 2:
        return text * 2  # "a" -> "aa", "" -> ""
    return text[0] + text[-1]


def hours_to_seconds(hours: float) -> float:
    """Converts a value in hours to seconds."""
    return hours * 3600


def get_modulo(dividend: int, divisor: int) -> int:
    """Returns the modulo of two given numbers."""
    if divisor == 0:
        raise ValueError("Cannot divide by zero.")
    return dividend % divisor


def max_triangle_third_edge(side1: int, side2: int) -> int:
    """
    Finds the maximum possible integer length of a triangle's third edge.
    The triangle inequality theorem states a + b > c.

    Args:
        side1: Length of the first side.
        side2: Length of the second side.

    Returns:
        The maximum integer length for the third side.
    """
    return (side1 + side2) - 1


def difference_max_min(numbers: List[float]) -> float:
    """Takes a list and returns the difference between the largest and smallest numbers."""
    if not numbers:
        return 0
    return max(numbers) - min(numbers)


def calculate_frame_count(minutes: float, fps: int) -> int:
    """Returns the number of frames shown in a given number of minutes for a certain FPS."""
    return minutes * 60 * fps


def is_string_empty(s: str) -> bool:
    """Returns True if a string is empty, False otherwise."""
    return not s


def inches_to_feet(inches: float) -> float:
    """Converts a measurement in inches to the equivalent in feet."""
    return inches / 12


def age_in_days(years: int) -> int:
    """
    Takes an age in years and returns the approximate age in days.
    Note: This is a simplification and does not account for leap years.
    """
    return years * 365


# ==============================================================================
# Section: Mathematical Sequences
# ==============================================================================


def sylvesters_sequence(n: int) -> int:
    """
    Calculates the nth term of Sylvester's sequence (iterative).
    Sequence: a(n) = a(n-1)^2 - a(n-1) + 1, with a(1) = 2.

    Args:
        n: The term number (must be >= 1).

    Returns:
        The nth term of the sequence.
    """
    if n < 1:
        raise ValueError("Sequence is defined for n >= 1.")
    a = 2
    for _ in range(1, n):
        a = a**2 - a + 1
    return a


def tribonacci_sequence(n: int) -> int:
    """
    Calculates the nth term of the Tribonacci sequence (iterative).
    Sequence: T(n) = T(n-1) + T(n-2) + T(n-3), with T(0)=0, T(1)=1, T(2)=1.

    Args:
        n: The term number (must be >= 0).

    Returns:
        The nth Tribonacci number.
    """
    if n < 0:
        raise ValueError("Sequence is not defined for negative numbers.")
    a, b, c = 0, 1, 1
    if n == 0: return a
    if n == 1: return b
    if n == 2: return c
    for _ in range(3, n + 1):
        a, b, c = b, c, a + b + c
    return c


def pell_sequence(n: int) -> int:
    """
    Calculates the nth term of the Pell sequence (iterative).
    Sequence: P(n) = 2*P(n-1) + P(n-2), with P(0)=0, P(1)=1.

    Args:
        n: The term number (must be >= 0).

    Returns:
        The nth Pell number.
    """
    if n < 0:
        raise ValueError("Sequence is not defined for negative numbers.")
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, 2 * b + a
    return b


def fermat_number(n: int) -> int:
    """
    Calculates the nth Fermat number.
    Formula: F(n) = 2^(2^n) + 1, for n >= 0.
    """
    if n < 0:
        raise ValueError("Sequence is not defined for negative numbers.")
    return 2**(2**n) + 1


def padovan_sequence(n: int) -> int:
    """
    Calculates the nth term of the Padovan sequence (iterative).
    Sequence: P(n) = P(n-2) + P(n-3), with P(0)=P(1)=P(2)=1.
    """
    if n < 0:
        raise ValueError("Sequence is not defined for negative numbers.")
    if n <= 2:
        return 1
    a, b, c = 1, 1, 1
    for _ in range(3, n + 1):
        a, b, c = b, c, a + b
    return c


def jacobsthal_number(n: int) -> int:
    """
    Calculates the nth Jacobsthal number (iterative).
    Sequence: J(n) = J(n-1) + 2*J(n-2), with J(0)=0, J(1)=1.
    """
    if n < 0:
        raise ValueError("Sequence is not defined for negative numbers.")
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, b + 2 * a
    return b


def perrin_number(n: int) -> int:
    """
    Calculates the nth Perrin number (iterative).
    Sequence: P(n) = P(n-2) + P(n-3), with P(0)=3, P(1)=0, P(2)=2.
    """
    if n < 0:
        raise ValueError("Sequence is not defined for negative numbers.")
    if n == 0: return 3
    if n == 1: return 0
    if n == 2: return 2
    a, b, c = 3, 0, 2
    for _ in range(3, n + 1):
        a, b, c = b, c, a + b
    return c


def cullen_number(n: int) -> int:
    """Calculates the nth Cullen number: C(n) = n * 2^n + 1."""
    if n < 0:
        raise ValueError("Sequence is not defined for negative numbers.")
    return n * (2**n) + 1


def woodall_number(n: int) -> int:
    """Calculates the nth Woodall number: W(n) = n * 2^n - 1."""
    if n < 1:
        raise ValueError("Sequence is typically defined for n >= 1.")
    return n * (2**n) - 1


def carol_number(n: int) -> int:
    """Calculates the nth Carol number: (2^n - 1)^2 - 2."""
    if n < 1:
        raise ValueError("Sequence is typically defined for n >= 1.")
    return (2**n - 1)**2 - 2


def star_number(n: int) -> int:
    """Calculates the nth star number: S(n) = 6n(n-1) + 1."""
    return 6 * n * (n - 1) + 1


def stella_octangula_number(n: int) -> int:
    """Calculates the nth Stella Octangula number: n * (2n^2 - 1)."""
    if n < 0:
        raise ValueError("Sequence is not defined for negative numbers.")
    return n * (2 * n**2 - 1)


# ==============================================================================
# Section: List & Data Structure Manipulations
# ==============================================================================


def remove_empty_tuples(tuples_list: List[Tuple]) -> List[Tuple]:
    """Removes empty tuples from a list of tuples."""
    return [t for t in tuples_list if t]


def count_occurrences(items: List[Any], element: Any) -> int:
    """Counts the number of occurrences of an element in a list."""
    return items.count(element)


def clone_list(original_list: List[Any]) -> List[Any]:
    """Creates a shallow copy of a list."""
    return list(original_list)


def get_odd_numbers(numbers: List[int]) -> List[int]:
    """Filters a list to return only the odd numbers."""
    return [num for num in numbers if num % 2 != 0]


def get_even_numbers(numbers: List[int]) -> List[int]:
    """Filters a list to return only the even numbers."""
    return [num for num in numbers if num % 2 == 0]


def find_n_largest(numbers: List[float], n: int) -> List[float]:
    """
    Finds the N largest elements from a list of numbers.

    Args:
        numbers: The list of numbers.
        n: The number of largest elements to find.

    Returns:
        A list containing the N largest elements, sorted.
    """
    if n <= 0:
        return []
    return heapq.nlargest(n, numbers)


def find_second_largest(numbers: List[float]) -> Optional[float]:
    """
    Finds the second largest number in a list.

    Returns:
        The second largest number, or None if the list has fewer than 2
        unique elements.
    """
    if len(numbers) < 2:
        return None

    unique_sorted_numbers = sorted(list(set(numbers)))
    if len(unique_sorted_numbers) < 2:
        return None

    return unique_sorted_numbers[-2]


def swap_first_last(items: List[Any]) -> List[Any]:
    """
    Swaps the first and last elements of a list in-place.

    Args:
        items: The list to be modified.

    Returns:
        The modified list.
    """
    if len(items) >= 2:
        items[0], items[-1] = items[-1], items[0]
    return items


@functools.lru_cache(maxsize=None)
def insertion_sort(arr: List[Any]) -> List[Any]:
    """
    Sorts a list using the Insertion Sort algorithm.

    Args:
        arr: The list to sort.

    Returns:
        A new sorted list.
    """
    arr_copy = arr[:]  # Work on a copy to avoid modifying the original
    for i in range(1, len(arr_copy)):
        key = arr_copy[i]
        j = i - 1
        while j >= 0 and key < arr_copy[j]:
            arr_copy[j + 1] = arr_copy[j]
            j -= 1
        arr_copy[j + 1] = key
    return arr_copy


def bubble_sort(arr: List[Any]) -> List[Any]:
    """
    Sorts a list using the Bubble Sort algorithm.

    Args:
        arr: The list to sort.

    Returns:
        A new sorted list.
    """
    arr_copy = arr[:]
    n = len(arr_copy)
    for i in range(n - 1):
        for j in range(0, n - i - 1):
            if arr_copy[j] > arr_copy[j + 1]:
                arr_copy[j], arr_copy[j + 1] = arr_copy[j + 1], arr_copy[j]
    return arr_copy


def selection_sort(arr: List[Any]) -> List[Any]:
    """
    Sorts a list using the Selection Sort algorithm.

    Args:
        arr: The list to sort.

    Returns:
        A new sorted list.
    """
    arr_copy = arr[:]
    for i in range(len(arr_copy)):
        min_idx = i
        for j in range(i + 1, len(arr_copy)):
            if arr_copy[min_idx] > arr_copy[j]:
                min_idx = j
        arr_copy[i], arr_copy[min_idx] = arr_copy[min_idx], arr_copy[i]
    return arr_copy


def get_cumulative_sum(numbers: List[float]) -> List[float]:
    """
    Calculates the cumulative sum of a list.

    Args:
        numbers: A list of numbers.

    Returns:
        A list where each element is the sum of all preceding elements
        in the input list, including itself.
    """
    return list(itertools.accumulate(numbers))


def chunk_list(data: List[Any], size: int) -> List[List[Any]]:
    """
    Breaks a list into smaller chunks of a specified size.

    Args:
        data: The list to chunk.
        size: The desired size of each chunk.

    Returns:
        A list of lists, where each sublist is a chunk.
    """
    if size <= 0:
        raise ValueError("Chunk size must be positive.")
    return [data[i:i + size] for i in range(0, len(data), size)]


def flatten_nested_list(nested_list: List[Any]) -> Generator[Any, None, None]:
    """
    Deeply flattens a nested list of arbitrary depth.

    Args:
        nested_list: The list to flatten, which may contain other lists.

    Returns:
        A generator that yields the flattened items.
    """
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_nested_list(item)
        else:
            yield item


# ==============================================================================
# Section: String Manipulations
# ==============================================================================


def add_binary_strings(bin_str1: str, bin_str2: str) -> str:
    """Adds two binary numbers represented as strings."""
    sum_int = int(bin_str1, 2) + int(bin_str2, 2)
    return bin(sum_int)[2:]  # [2:] to remove the '0b' prefix


def count_vowels(input_string: str) -> Dict[str, int]:
    """
    Counts the number of occurrences of each vowel in a string.

    Args:
        input_string: The string to analyze.

    Returns:
        A dictionary with vowels as keys and their counts as values.
    """
    vowels = "aeiou"
    counter = collections.Counter(input_string.lower())
    return {vowel: counter[vowel] for vowel in vowels}


def sort_words_alphabetically(sentence: str) -> List[str]:
    """Sorts the words from a string alphabetically (case-insensitive)."""
    words = sentence.lower().split()
    words.sort()
    return words


def remove_punctuation(input_string: str) -> str:
    """
    Removes all punctuation characters from a string.

    Returns:
        The string with punctuation removed.
    """
    # Create a translation table to remove punctuation
    translator = str.maketrans('', '', string.punctuation)
    return input_string.translate(translator)


def is_palindrome(input_string: str) -> bool:
    """
    Checks if a string is a palindrome (reads the same forwards and backwards),
    case-insensitively.
    """
    processed_str = input_string.casefold()
    return processed_str == processed_str[::-1]


def to_sentence_case(paragraph: str) -> str:
    """
    Converts a paragraph string into sentence case.
    Capitalizes the first letter of each sentence.
    """
    sentences = paragraph.split('. ')
    capitalized = [s.capitalize() for s in sentences]
    return '. '.join(capitalized)


def string_to_int_list(s: str) -> List[int]:
    """Converts a space-separated string of numbers into a list of integers."""
    return [int(num) for num in s.split()]


def is_substring_present(main_string: str, substring: str) -> bool:
    """Checks if a substring is present in a given string."""
    return substring in main_string


def get_word_frequencies(text: str) -> Dict[str, int]:
    """Calculates the frequency of each word in a string."""
    return collections.Counter(text.split())


def snake_to_pascal(snake_case_string: str) -> str:
    """Converts a snake_case string to PascalCase."""
    return snake_case_string.replace("_", " ").title().replace(" ", "")


def get_even_length_words(text: str) -> List[str]:
    """Returns a list of words with even length from a string."""
    return [word for word in text.split() if len(word) % 2 == 0]


def remove_duplicate_chars(text: str) -> str:
    """
    Removes all duplicate characters from a string while preserving order.
    'geeksforgeeks' -> 'geksfor'
    """
    # dict.fromkeys preserves insertion order (Python 3.7+)
    return "".join(dict.fromkeys(text))


def get_least_frequent_char(text: str) -> Optional[str]:
    """Finds the least frequent character in a string."""
    if not text:
        return None
    counts = collections.Counter(text)
    # most_common() returns list of (element, count) tuples
    return counts.most_common()[-1][0]


def get_most_frequent_char(text: str) -> Optional[str]:
    """Finds the most frequent character in a string."""
    if not text:
        return None
    counts = collections.Counter(text)
    return counts.most_common(1)[0][0]


def filter_words_by_length(text: str, k: int) -> List[str]:
    """Finds all words in a string that are longer than a given length k."""
    return [word for word in text.split() if len(x) > k]


def is_binary_string(text: str) -> bool:
    """Checks if a string contains only '0's and '1's."""
    return all(char in '01' for char in text)


def find_uncommon_words(text_a: str, text_b: str) -> List[str]:
    """
    Finds a list of uncommon words between two strings.
    An uncommon word appears exactly once in total across both strings.
    """
    counts = collections.Counter((text_a + " " + text_b).split())
    return [word for word, count in counts.items() if count == 1]


def rotate_string(text: str, d: int) -> Tuple[str, str]:
    """
    Rotates a string left and right by d positions.

    Args:
        text: The string to rotate.
        d: The number of positions to rotate by.

    Returns:
        A tuple containing (left_rotated_string, right_rotated_string).
    """
    d = d % len(text)
    left_rotated = text[d:] + text[:d]
    right_rotated = text[-d:] + text[:-d]
    return (left_rotated, right_rotated)


# ==============================================================================
# Section: Dictionary & Matrix Manipulations
# ==============================================================================


def transpose_matrix(matrix: List[List[Any]]) -> List[List[Any]]:
    """
    Transposes a matrix (swaps rows and columns).

    Args:
        matrix: A 2D list representing the matrix.

    Returns:
        The transposed matrix.
    """
    if not matrix or not matrix[0]:
        return []
    return [list(row) for row in zip(*matrix)]


def invert_dictionary(
    d: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Swaps keys and values in a dictionary.

    Note: This requires all values in the original dictionary to be unique
    and hashable.

    Args:
        d: The dictionary to invert.

    Returns:
        The inverted dictionary.
    """
    if len(d.values()) != len(set(d.values())):
        raise ValueError("Dictionary values must be unique to be inverted.")
    return {value: key for key, value in d.items()}


def get_unique_values_from_dict(d: Dict[Any, List[Any]]) -> List[Any]:
    """Extracts all unique values from a dictionary whose values are lists."""
    all_values = set()
    for value_list in d.values():
        all_values.update(value_list)
    return sorted(list(all_values))


def sum_dictionary_values(d: Dict[Any, float]) -> float:
    """Finds the sum of all numeric values in a dictionary."""
    return sum(d.values())


def merge_dictionaries(dict1: Dict, dict2: Dict) -> Dict:
    """Merges two dictionaries. Values from the second dict overwrite the first."""
    return {**dict1, **dict2}


# ==============================================================================
# Section: Advanced & Algorithmic
# ==============================================================================


def simple_interest(principal: float, time: float, rate: float) -> float:
    """Calculates simple interest."""
    return (principal * time * rate) / 100


def get_factors(n: int) -> List[int]:
    """
    Finds all factors of a given number.

    Args:
        n: The number to factorize (must be positive).

    Returns:
        A list of its factors.
    """
    if n <= 0:
        raise ValueError("Input must be a positive integer.")
    return [i for i in range(1, n + 1) if n % i == 0]


def compute_hcf(x: int, y: int) -> int:
    """Computes the Highest Common Factor (HCF/GCD) of two numbers."""
    return math.gcd(x, y)


def compute_lcm(x: int, y: int) -> int:
    """Computes the Lowest Common Multiple (LCM) of two numbers."""
    if x == 0 or y == 0:
        return 0
    return abs(x * y) // math.gcd(x, y)


def is_armstrong_number(n: int) -> bool:
    """Checks if a number is an Armstrong number."""
    if n < 0:
        return False
    s_n = str(n)
    order = len(s_n)
    return n == sum(int(digit)**order for digit in s_n)


def find_armstrong_in_range(lower: int, upper: int) -> List[int]:
    """Finds all Armstrong numbers within a given interval."""
    return [num for num in range(lower, upper + 1) if is_armstrong_number(num)]


def is_prime(n: int) -> bool:
    """
    Checks if a number is prime using an optimized trial division method.

    Args:
        n: The number to check.

    Returns:
        True if the number is prime, False otherwise.
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


# Note: A class-based example for caching properties
class PropertyCache:
    """
    A decorator to cache a property's result after the first access.
    This is useful for computationally expensive properties.
    Note: Python 3.8+ offers `functools.cached_property` which is preferred.
    """
    def __init__(self, func: Callable):
        self.func = func
        self.func_name = func.__name__

    def __get__(self, obj, cls):
        if obj is None:
            return self
        value = self.func(obj)
        setattr(obj, self.func_name, value)
        return value


class TowerOfHanoi:
    """A class to solve the Tower of Hanoi puzzle."""
    def __init__(self):
        self.moves = []

    def solve(self, n: int, source: str, destination: str, auxiliary: str):
        """Generates the moves to solve the puzzle."""
        if n == 1:
            self.moves.append(
                f"Move disk 1 from {source} to {destination}")
            return
        self.solve(n - 1, source, auxiliary, destination)
        self.moves.append(f"Move disk {n} from {source} to {destination}")
        self.solve(n - 1, auxiliary, destination, source)

    def get_moves(self, n: int, s='A', d='B', a='C') -> List[str]:
        """Returns the list of moves for n disks."""
        self.moves = []
        self.solve(n, s, d, a)
        return self.moves


# ==============================================================================
# Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    print("--- Running Refactored Python Examples ---")

    # 74: ASCII value
    print(f"\n[74] ASCII value of 'A' is: {get_ascii_value('A')}")

    # 75: Union of sets
    set_a = {1, 2, 3}
    set_b = {3, 4, 5}
    print(
        f"\n[75] Union of {set_a} and {set_b} is: {union_of_sets(set_a, set_b)}"
    )

    # 76: Add member to a set
    color_set = {"Blue"}
    color_set.add("Red")
    print(f"\n[76] Set after adding 'Red': {color_set}")

    # 77: Add two lists
    list_1 = [1, 2, 3]
    list_2 = [4, 5, 6]
    print(f"\n[77] Sum of {list_1} and {list_2}: {add_lists_elementwise(list_1, list_2)}")

    # 78: Divisible by digits
    print(f"\n[78] Numbers between 1 and 22 divisible by their digits: {find_numbers_divisible_by_digits(1, 22)}")

    # 96: Difference between max and min
    num_list = [10, 4, 25, 1, 99]
    print(
        f"\n[96] Difference between max and min of {num_list} is: {difference_max_min(num_list)}"
    )

    # 100: Age in days
    print(f"\n[100] 30 years is approximately {age_in_days(30)} days.")

    # Remove empty tuples from a list
    tuples = [(), ('ram', '15', '8'), (), ('laxman', 'sita')]
    print(
        f"\n[List] Removing empty tuples from {tuples}: {remove_empty_tuples(tuples)}"
    )

    # Count occurrences in a list
    my_list = [8, 6, 8, 10, 8, 20, 10, 8, 8]
    element_to_count = 8
    print(
        f"\n[List] The element {element_to_count} appears {count_occurrences(my_list, element_to_count)} times in {my_list}"
    )

    # Find N largest elements
    n_largest_list = [1000, 298, 3579, 100, 200, -45, 900]
    n = 4
    print(
        f"\n[List] The {n} largest elements in {n_largest_list} are: {find_n_largest(n_largest_list, n)}"
    )

    # Find second largest number
    second_largest_list = [10, 20, 4, 45, 99, 99]
    print(
        f"\n[List] Second largest in {second_largest_list}: {find_second_largest(second_largest_list)}"
    )

    # Swap first and last element of a list
    swap_me = [12, 35, 9, 56, 24]
    print(f"\n[List] Swapping first/last of {swap_me}: {swap_first_last(swap_me[:])}")

    # Simple Interest
    p, t, r = 1000, 2, 5  # Principal, Time, Rate
    print(
        f"\n[Math] Simple interest for P={p}, T={t}, R={r}% is: {simple_interest(p, t, r)}"
    )

    # Sorting algorithms
    unsorted_array = [64, 34, 25, 12, 22, 11, 90]
    print(f"\n[Sort] Bubble Sort of {unsorted_array}: {bubble_sort(unsorted_array)}")
    print(f"[Sort] Insertion Sort of {unsorted_array}: {insertion_sort(unsorted_array)}")
    print(f"[Sort] Selection Sort of {unsorted_array}: {selection_sort(unsorted_array)}")

    # Add binary strings
    print(f"\n[String] Binary '101' + '11' = {add_binary_strings('101', '11')}")

    # Count vowels
    vowel_test_str = 'Hello, have you tried our tutorial section yet?'
    print(f"\n[String] Vowel counts in '{vowel_test_str}': {count_vowels(vowel_test_str)}")

    # Remove punctuation
    punct_str = "Hello!!!, he said ---and went."
    print(
        f"\n[String] Removing punctuation from '{punct_str}': {remove_punctuation(punct_str)}"
    )

    # Palindrome check
    palindrome_str = 'A man, a plan, a canal: Panama'
    clean_palindrome_str = ''.join(filter(str.isalnum, palindrome_str))
    print(
        f"\n[String] Is '{clean_palindrome_str}' a palindrome? {is_palindrome(clean_palindrome_str)}"
    )

    # Transpose Matrix
    matrix = [[12, 7], [4, 5], [3, 8]]
    print(f"\n[Matrix] Transpose of {matrix} is {transpose_matrix(matrix)}")

    # HCF and LCM
    num1, num2 = 54, 24
    print(f"\n[Math] HCF of {num1}, {num2} is: {compute_hcf(num1, num2)}")
    print(f"[Math] LCM of {num1}, {num2} is: {compute_lcm(num1, num2)}")

    # Armstrong Number
    print(f"\n[Math] Is 153 an Armstrong number? {is_armstrong_number(153)}")
    print(f"[Math] Armstrong numbers between 100 and 1000: {find_armstrong_in_range(100, 1000)}")

    # Prime Number
    print(f"\n[Math] Is 97 a prime number? {is_prime(97)}")

    # Flatten a list
    nested = [1, 2, [3, 4], [5, 6, [7]]]
    print(
        f"\n[List] Flattening {nested}: {list(flatten_nested_list(nested))}"
    )

    # Merge Dictionaries
    dict_x = {'a': 1, 'b': 2}
    dict_y = {'b': 3, 'c': 4}
    print(
        f"\n[Dict] Merging {dict_x} and {dict_y}: {merge_dictionaries(dict_x, dict_y)}"
    )

    # Word Frequencies
    freq_text = 'Gfg is best Geeks are good and Geeks like Gfg'
    print(f"\n[String] Word frequencies: {get_word_frequencies(freq_text)}")

    # Snake to Pascal Case
    snake_str = 'geeksforgeeks_is_best'
    print(f"\n[String] '{snake_str}' in PascalCase: {snake_to_pascal(snake_str)}")

    # Tower of Hanoi
    hanoi_solver = TowerOfHanoi()
    moves = hanoi_solver.get_moves(n=3)
    print("\n[Algo] Tower of Hanoi for 3 disks:")
    for move in moves:
        print(f"  - {move}")

    print("\n--- End of Examples ---")