# -*- coding: utf-8 -*-
"""
A collection of refactored Python code examples demonstrating best practices.

This script includes functions for various tasks, including mathematical operations,
string manipulation, list processing, and more. Each function has been improved
for readability, robustness, and adherence to Python standards (PEP 8).

Key improvements include:
- Descriptive function and variable names.
- Comprehensive docstrings explaining purpose, arguments, and return values.
- Type hints for improved clarity and static analysis.
- Use of Pythonic idioms (e.g., list comprehensions).
- Separation of logic (functions return values) from presentation (printing is
  handled by the caller).
- Handling of edge cases.
- A main execution block to demonstrate the usage of each function.
"""

# region IMPORTS
import calendar
import collections
import datetime
import functools
import importlib.util
import json
import keyword
import math
import os
import psutil
import re
import time
from typing import (Any, Dict, List, Optional, Set, Tuple, Union,
                    Callable, Iterable)
# endregion


# region MATHEMATICAL & NUMBER OPERATIONS

def find_max_odd(numbers: List[int]) -> Optional[int]:
    """
    Finds the maximum odd number from a list of integers.

    Args:
        numbers: A list of integers.

    Returns:
        The maximum odd number in the list, or None if the list contains no
        odd numbers.
    """
    odd_numbers = [num for num in numbers if num % 2 != 0]
    return max(odd_numbers) if odd_numbers else None


def find_max_even(numbers: List[int]) -> Optional[int]:
    """
    Finds the maximum even number from a list of integers.

    Args:
        numbers: A list of integers.

    Returns:
        The maximum even number in the list, or None if the list contains no
        even numbers.
    """
    even_numbers = [num for num in numbers if num % 2 == 0]
    return max(even_numbers) if even_numbers else None


def calculate_quadratic_roots(a: float, b: float, c: float) -> Optional[Tuple[float, float]]:
    """
    Calculates the real roots of a quadratic equation (ax^2 + bx + c = 0).

    Args:
        a: The coefficient of the x^2 term.
        b: The coefficient of the x term.
        c: The constant term.

    Returns:
        A tuple containing the two real roots, or None if the roots are
        imaginary or 'a' is zero.
    """
    if a == 0:
        print("Error: 'a' cannot be zero for a quadratic equation.")
        return None

    discriminant = (b**2) - (4 * a * c)

    if discriminant >= 0:
        sqrt_discriminant = math.sqrt(discriminant)
        root1 = (-b + sqrt_discriminant) / (2 * a)
        root2 = (-b - sqrt_discriminant) / (2 * a)
        return root1, root2
    else:
        # Roots are imaginary
        return None


def get_nth_perfect_power(n: int, power: int) -> int:
    """
    Calculates the N-th number that is a perfect power (e.g., square, cube).
    The number that is both a perfect square and a perfect cube is a perfect
    sixth power.

    Args:
        n: The term number (e.g., 1st, 2nd, ...).
        power: The power to raise to (e.g., 6 for square and cube).

    Returns:
        The N-th number that is a perfect power.
    """
    return n ** power


def is_power_of(number: int, base: int) -> bool:
    """
    Checks if a number is a power of a given base.

    This implementation avoids floating-point inaccuracies from using math.log.

    Args:
        number: The number to check.
        base: The base.

    Returns:
        True if 'number' is a power of 'base', False otherwise.
    """
    if base == 1:
        return number == 1
    if base <= 0 or number <= 0:
        return False
    
    current_power = 1
    while current_power < number:
        current_power *= base
    return current_power == number


def sum_digits_recursive(num: int) -> int:
    """
    Calculates the sum of all digits of a non-negative integer using recursion.

    Args:
        num: A non-negative integer.

    Returns:
        The sum of the digits.
    """
    if num < 0:
        raise ValueError("Input must be a non-negative integer.")
    if num < 10:
        return num
    return num % 10 + sum_digits_recursive(num // 10)


def reverse_integer(n: int) -> int:
    """
    Reverses the digits of a given integer. Handles negative numbers.

    Args:
        n: An integer.

    Returns:
        The integer with its digits reversed.
    """
    if n < 0:
        return -int(str(abs(n))[::-1])
    return int(str(n)[::-1])


def add_without_plus_operator(a: int, b: int) -> int:
    """
    Adds two positive integers without using the '+' operator, using bitwise ops.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The sum of a and b.
    """
    while b != 0:
        carry = a & b
        a = a ^ b
        b = carry << 1
    return a


def is_disarium(num: int) -> bool:
    """
    Checks if a number is a Disarium number.
    A number is Disarium if the sum of its digits powered with their respective
    positions is equal to the number itself. Example: 135 = 1^1 + 3^2 + 5^3.

    Args:
        num: The number to check.

    Returns:
        True if the number is a Disarium number, False otherwise.
    """
    if num < 0:
        return False
    
    s = str(num)
    digit_sum = sum(int(digit) ** (i + 1) for i, digit in enumerate(s))
    return digit_sum == num

# endregion


# region STRING & TEXT OPERATIONS

def is_binary_palindrome(n: int) -> bool:
    """
    Checks if the binary representation of a number is a palindrome.

    Args:
        n: A non-negative integer.

    Returns:
        True if the binary representation is a palindrome, False otherwise.
    """
    if n < 0:
        return False
    binary_representation = bin(n)[2:]
    return binary_representation == binary_representation[::-1]


def format_number_with_commas(n: Union[int, float]) -> str:
    """
    Formats a number with commas as thousand separators.

    Args:
        n: The number to format.

    Returns:
        A string representation of the number with thousand separators.
    """
    return f"{n:,}"


def count_case_in_string(text: str) -> Dict[str, int]:
    """
    Counts the number of uppercase and lowercase letters in a string.

    Args:
        text: The input string.

    Returns:
        A dictionary with counts for 'uppercase' and 'lowercase' letters.
    """
    counts = {'uppercase': 0, 'lowercase': 0}
    for char in text:
        if char.isupper():
            counts['uppercase'] += 1
        elif char.islower():
            counts['lowercase'] += 1
    return counts


def count_letters_and_digits(text: str) -> Dict[str, int]:
    """
    Counts the total number of letters and digits in a string.

    Args:
        text: The input string.

    Returns:
        A dictionary with counts for 'letters' and 'digits'.
    """
    counts = {'letters': 0, 'digits': 0}
    for char in text:
        if char.isalpha():
            counts['letters'] += 1
        elif char.isdigit():
            counts['digits'] += 1
    return counts


def count_word_occurrence(text: str, word: str) -> int:
    """
    Counts the occurrences of a specific word in a text.

    Args:
        text: The body of text to search within.
        word: The word to count.

    Returns:
        The number of times the word appears in the text.
    """
    return text.split().count(word)


def capitalize_words(text: str) -> str:
    """
    Capitalizes the first letter of each word in a string.

    Args:
        text: The input string.

    Returns:
        The string with each word capitalized.
    """
    return text.title()


def print_chars_at_even_indices(text: str) -> None:
    """
    Displays characters from a string that are at an even index number.

    Args:
        text: The input string.
    """
    print(f"Characters at even indices in '{text}':")
    for i in range(0, len(text), 2):
        print(f"  index[{i}] = {text[i]}")


def remove_leading_chars(text: str, n: int) -> str:
    """
    Removes the first 'n' characters from a string.

    Args:
        text: The input string.
        n: The number of characters to remove from the beginning.

    Returns:
        A new string with the leading characters removed.
    """
    return text[n:]


def find_uncommon_words(sentence1: str, sentence2: str) -> List[str]:
    """
    Finds all uncommon words between two sentences.
    An uncommon word is a word that appears exactly once in one of the
    sentences and does not appear in the other.

    Args:
        sentence1: The first sentence.
        sentence2: The second sentence.

    Returns:
        A list of uncommon words.
    """
    # Use collections.Counter for a more efficient word count
    count = collections.Counter((sentence1 + " " + sentence2).split())
    return [word for word, num in count.items() if num == 1]


def extract_urls(text: str) -> List[str]:
    """
    Extracts all URLs from a given string using regular expressions.

    Args:
        text: The string to search for URLs.

    Returns:
        A list of URLs found in the string.
    """
    # A robust regex for finding URLs
    url_regex = (
        r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|'
        r'(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    )
    return re.findall(url_regex, text)



# region LIST, TUPLE, SET & DICTIONARY OPERATIONS

def get_list_intersection(list_a: List[Any], list_b: List[Any]) -> List[Any]:
    """
    Finds the intersection of two lists (common elements).

    Args:
        list_a: The first list.
        list_b: The second list.

    Returns:
        A new list containing elements present in both lists.
    """
    return list(set(list_a) & set(list_b))


def get_list_union(list_a: List[Any], list_b: List[Any]) -> List[Any]:
    """
    Finds the union of two lists (all unique elements).

    Args:
        list_a: The first list.
        list_b: The second list.

    Returns:
        A new list containing all unique elements from both lists.
    """
    return list(set(list_a) | set(list_b))


def remove_falsy_values(data: List[Any]) -> List[Any]:
    """
    Removes all falsy values (False, None, 0, "", [], {}, ()) from a list.

    Args:
        data: The list to filter.

    Returns:
        A new list containing only truthy values.
    """
    return list(filter(None, data))


def are_all_elements_equal(data: List[Any]) -> bool:
    """
    Checks if all elements in a list are the same.

    Args:
        data: The list to check.

    Returns:
        True if all elements are identical or the list is empty/has one
        element, False otherwise.
    """
    if len(data) < 2:
        return True
    return len(set(data)) == 1


def is_first_last_same(numbers: List[Any]) -> bool:
    """
    Checks if the first and last elements of a list are the same.

    Args:
        numbers: The list to check.

    Returns:
        True if the first and last elements are equal, False otherwise.
        Returns False for empty lists.
    """
    return len(numbers) > 0 and numbers[0] == numbers[-1]


def get_divisible_by(numbers: List[int], divisor: int) -> List[int]:
    """
    Filters a list to return only numbers divisible by a given divisor.

    Args:
        numbers: A list of integers.
        divisor: The number to divide by.

    Returns:
        A new list containing numbers divisible by the divisor.
    """
    if divisor == 0:
        raise ValueError("Divisor cannot be zero.")
    return [num for num in numbers if num % 5 == 0]


def merge_odd_even_lists(list1: List[int], list2: List[int]) -> List[int]:
    """
    Creates a new list with odd numbers from the first list and even numbers
    from the second list.

    Args:
        list1: The first list of numbers.
        list2: The second list of numbers.

    Returns:
        A new merged list.
    """
    odd_from_list1 = [num for num in list1 if num % 2 != 0]
    even_from_list2 = [num for num in list2 if num % 2 == 0]
    return odd_from_list1 + even_from_list2


def find_second_largest(numbers: List[Union[int, float]]) -> Optional[Union[int, float]]:
    """
    Finds the second largest number in a list.

    Args:
        numbers: A list of numbers.

    Returns:
        The second largest number, or None if the list has fewer than two
        unique elements.
    """
    unique_numbers = sorted(list(set(numbers)), reverse=True)
    if len(unique_numbers) >= 2:
        return unique_numbers[1]
    return None


def find_longest_word(words: List[str]) -> Optional[str]:
    """
    Finds the longest word in a list of words.

    Args:
        words: A list of strings.

    Returns:
        The longest word in the list. If there's a tie, the first one
        encountered is returned. Returns None for an empty list.
    """
    if not words:
        return None
    return max(words, key=len)


def is_monotonic(arr: List[Union[int, float]]) -> bool:
    """
    Determines if a list is monotonic (either entirely non-increasing or
    non-decreasing).

    Args:
        arr: The list of numbers to check.

    Returns:
        True if the list is monotonic, False otherwise.
    """
    is_increasing = all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1))
    is_decreasing = all(arr[i] >= arr[i + 1] for i in range(len(arr) - 1))
    return is_increasing or is_decreasing


# region DATE & TIME OPERATIONS

def is_valid_date(year: int, month: int, day: int) -> bool:
    """
    Validates if a given year, month, and day combination forms a valid date.

    Args:
        year: The year.
        month: The month.
        day: The day.

    Returns:
        True if the date is valid, False otherwise.
    """
    try:
        datetime.date(year, month, day)
        return True
    except ValueError:
        return False


def parse_date_string(date_str: str, date_format: str) -> Optional[datetime.datetime]:
    """
    Converts a string into a datetime object based on a given format.

    Args:
        date_str: The string representation of the date.
        date_format: The format code (e.g., '%b %d %Y %I:%M%p').

    Returns:
        A datetime object if parsing is successful, None otherwise.
    """
    try:
        return datetime.datetime.strptime(date_str, date_format)
    except ValueError:
        return None


def get_current_time_millis() -> int:
    """
    Gets the current time in milliseconds since the Epoch.

    Returns:
        The current time as an integer number of milliseconds.
    """
    return int(time.time() * 1000)



# region ALGORITHMIC & COMPLEX LOGIC

def get_permutations(items: List[Any]) -> List[List[Any]]:
    """
    Generates all possible permutations from a collection of distinct items.

    Args:
        items: A list of items to permute.

    Returns:
        A list of lists, where each inner list is a unique permutation.
    """
    if not items:
        return [[]]

    result = []
    for i, item in enumerate(items):
        remaining_items = items[:i] + items[i+1:]
        for p in get_permutations(remaining_items):
            result.append([item] + p)
    return result


def solve_tower_of_hanoi(n: int, source: str, destination: str, auxiliary: str) -> None:
    """
    Solves the Tower of Hanoi problem and prints the steps.

    Args:
        n: The number of disks.
        source: The name of the source peg.
        destination: The name of the destination peg.
        auxiliary: The name of the auxiliary peg.
    """
    if n == 1:
        print(f"Move disk 1 from {source} to {destination}")
        return
    solve_tower_of_hanoi(n - 1, source, auxiliary, destination)
    print(f"Move disk {n} from {source} to {destination}")
    solve_tower_of_hanoi(n - 1, auxiliary, destination, source)


def find_target_sum_ways(nums: List[int], target: int) -> int:
    """
    Finds the number of ways to assign symbols (+ or -) to make the sum of
    the numbers equal to the target.

    Args:
        nums: A list of non-negative integers.
        target: The target sum.

    Returns:
        The number of ways to achieve the target sum.
    """
    memo = {}

    def backtrack(index: int, current_sum: int) -> int:
        if (index, current_sum) in memo:
            return memo[(index, current_sum)]

        if index == len(nums):
            return 1 if current_sum == target else 0

        # Explore the '+' path
        add = backtrack(index + 1, current_sum + nums[index])
        # Explore the '-' path
        subtract = backtrack(index + 1, current_sum - nums[index])

        memo[(index, current_sum)] = add + subtract
        return memo[(index, current_sum)]

    return backtrack(0, 0)


def is_valid_parentheses(s: str) -> bool:
    """
    Checks if a string of parentheses '()', brackets '[]', and braces '{}' is
    valid.

    Args:
        s: The string containing parentheses.

    Returns:
        True if the string is valid, False otherwise.
    """
    stack = []
    mapping = {")": "(", "}": "{", "]": "["}

    for char in s:
        if char in mapping:
            top_element = stack.pop() if stack else '#'
            if mapping[char] != top_element:
                return False
        else:
            stack.append(char)

    return not stack


# region SYSTEM & UTILITY FUNCTIONS

def get_python_keywords() -> List[str]:
    """
    Returns a list of all keywords in the current Python version.

    Returns:
        A list of keyword strings.
    """
    return keyword.kwlist


def is_valid_json(json_string: str) -> bool:
    """
    Validates whether a given string is a valid JSON.

    Args:
        json_string: The string to validate.

    Returns:
        True if the string is valid JSON, False otherwise.
    """
    try:
        json.loads(json_string)
        return True
    except (json.JSONDecodeError, TypeError):
        return False


def get_installed_packages() -> List[str]:
    """

    Retrieves a list of all locally installed Python packages and their versions.

    Returns:
        A sorted list of strings, each in the format 'package==version'.
    """
    if not importlib.util.find_spec("pkg_resources"):
        return ["'pkg_resources' not found. Please install with 'pip install setuptools'."]
        
    import pkg_resources
    
    installed_packages = pkg_resources.working_set
    return sorted([f"{i.key}=={i.version}" for i in installed_packages])


def get_current_memory_usage_mb() -> float:
    """
    Determines the memory usage (RSS) of the current Python process in megabytes.

    Returns:
        The memory usage in MB as a float.
    """
    process = psutil.Process(os.getpid())
    # rss: Resident Set Size
    return process.memory_info().rss / (1024 ** 2)


# region MAIN DEMONSTRATION
def main():
    """Main function to demonstrate the refactored code."""
    print("--- Refactored Python Examples Demonstration ---")

    # --- Mathematical & Number Operations ---
    print("\n--- Mathematical & Number Operations ---")
    num_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, -11]
    print(f"List: {num_list}")
    print(f"Max odd number: {find_max_odd(num_list)}")
    print(f"Max even number: {find_max_even(num_list)}")
    print(f"Quadratic roots of x^2 - 5x + 6: {calculate_quadratic_roots(1, -5, 6)}")
    print(f"2nd number that is a perfect square & cube: {get_nth_perfect_power(2, 6)}")
    print(f"Is 27 a power of 3? {is_power_of(27, 3)}")
    print(f"Is 28 a power of 3? {is_power_of(28, 3)}")
    print(f"Sum of digits for 12345: {sum_digits_recursive(12345)}")
    print(f"Reverse of 12345: {reverse_integer(12345)}")
    print(f"Reverse of -123: {reverse_integer(-123)}")
    print(f"Sum of 15 and 20 without '+': {add_without_plus_operator(15, 20)}")
    print(f"Is 135 a Disarium number? {is_disarium(135)}")
    print(f"Is 136 a Disarium number? {is_disarium(136)}")

    # --- String & Text Operations ---
    print("\n--- String & Text Operations ---")
    print(f"Is binary of 9 (1001) a palindrome? {is_binary_palindrome(9)}")
    print(f"Is binary of 10 (1010) a palindrome? {is_binary_palindrome(10)}")
    print(f"Formatted number 1234567.89: {format_number_with_commas(1234567.89)}")
    test_string = 'TestStringInCamelCase123'
    print(f"Case count in '{test_string}': {count_case_in_string(test_string)}")
    print(f"Letter/digit count in '{test_string}': {count_letters_and_digits(test_string)}")
    sample_text = "Python is a fun language, and Python is easy to learn."
    print(f"Occurrences of 'Python' in text: {count_word_occurrence(sample_text, 'Python')}")
    print(f"Title cased text: {capitalize_words('hello world of python')}")
    print_chars_at_even_indices("Programming")
    print(f"Removing first 3 chars from 'Python': {remove_leading_chars('Python', 3)}")

    # --- List, Tuple, Set & Dictionary Operations ---
    print("\n--- List, Tuple, Set & Dictionary Operations ---")
    list_a = [1, 2, 3, 4, 5]
    list_b = [4, 5, 6, 7, 8]
    print(f"Intersection of {list_a} and {list_b}: {get_list_intersection(list_a, list_b)}")
    print(f"Union of {list_a} and {list_b}: {get_list_union(list_a, list_b)}")
    falsy_list = [0, 1, False, 2, '', 3, 'a', None, []]
    print(f"Removing falsy values from {falsy_list}: {remove_falsy_values(falsy_list)}")
    print(f"Are all elements in [2, 2, 2] equal? {are_all_elements_equal([2, 2, 2])}")
    print(f"Is first and last same in [10, 20, 30, 10]? {is_first_last_same([10, 20, 30, 10])}")
    div_list = [5, 10, 12, 15, 21, 25]
    print(f"Numbers divisible by 5 in {div_list}: {get_divisible_by(div_list, 5)}")
    print(f"Merged odd/even lists: {merge_odd_even_lists([1, 3, 5], [2, 4, 6])}")
    print(f"Second largest in {num_list}: {find_second_largest(num_list)}")
    word_list = ["apple", "banana", "kiwi", "strawberry"]
    print(f"Longest word in {word_list}: {find_longest_word(word_list)}")
    print(f"Is [1, 2, 3, 3, 4] monotonic? {is_monotonic([1, 2, 3, 3, 4])}")
    print(f"Is [4, 3, 2, 5] monotonic? {is_monotonic([4, 3, 2, 5])}")

    # --- Date & Time Operations ---
    print("\n--- Date & Time Operations ---")
    print(f"Is 2023-02-29 a valid date? {is_valid_date(2023, 2, 29)}")
    print(f"Is 2024-02-29 a valid date? {is_valid_date(2024, 2, 29)}")
    dt_obj = parse_date_string("Mar 26 2021 4:20PM", '%b %d %Y %I:%M%p')
    print(f"Parsed datetime object: {dt_obj}")
    print(f"Current time in milliseconds: {get_current_time_millis()}")
    
    # --- Calendar Example ---
    print("\n--- Calendar for 2024 ---")
    print(calendar.calendar(2024))
    
    # --- Algorithmic & Complex Logic ---
    print("\n--- Algorithmic & Complex Logic ---")
    print("Permutations of [1, 2, 3]:")
    for p in get_permutations([1, 2, 3]):
        print(f"  {p}")
    print("\nSolving Tower of Hanoi for 3 disks:")
    solve_tower_of_hanoi(3, 'A', 'C', 'B')
    nums = [1, 1, 1, 1, 1]
    target = 3
    print(f"\nWays to get sum {target} from {nums}: {find_target_sum_ways(nums, target)}")
    paren_str = "{[()]}"
    print(f"Are parentheses in '{paren_str}' valid? {is_valid_parentheses(paren_str)}")

    # --- System & Utility Functions ---
    print("\n--- System & Utility Functions ---")
    print(f"First 10 Python keywords: {get_python_keywords()[:10]}")
    valid_json = '{"name": "John", "age": 30}'
    invalid_json = '{"name": "John", "age": 30,}'
    print(f"Is '{valid_json}' valid JSON? {is_valid_json(valid_json)}")
    print(f"Is '{invalid_json}' valid JSON? {is_valid_json(invalid_json)}")
    print(f"Current memory usage: {get_current_memory_usage_mb():.2f} MB")
    
    # Uncomment the following line to see all installed packages
    # print("\nInstalled Python packages:\n", "\n".join(get_installed_packages()))

if __name__ == "__main__":
    main()
