"""
A collection of refactored Python code examples demonstrating best practices.

This script covers various topics including:
- JSON manipulation
- List and string operations
- Random data generation
- Date and time handling
- Common algorithms and data structures
- File I/O and system interactions

Each example is encapsulated in a well-documented function with type hints.
A `main()` function at the end demonstrates the usage of each function.
"""

# Step 1: Group all imports at the top of the file.
# Standard library imports
import json
import random
import re
import string
import sys
import time
import uuid
import os
import math
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from functools import reduce

# Third-party imports (if any, like numpy)
# Note: Ensure numpy is installed (`pip install numpy`)
import numpy as np
import requests

# Constants can be defined at the module level for clarity.
DATE_FORMAT = '%m/%d/%Y'

# ==============================================================================
# SECTION 1: JSON Operations
# ==============================================================================

def pretty_print_json(data: dict) -> str:
    """
    Converts a dictionary to a nicely formatted JSON string.

    Args:
        data: The dictionary to convert.

    Returns:
        A pretty-printed JSON string with an indent of 4 and sorted keys.
    """
    return json.dumps(data, indent=4, sort_keys=True)


def check_key_in_json_data(data: dict, key: str) -> bool:
    """
    Checks if a given key exists in a dictionary (parsed from JSON).

    Args:
        data: The dictionary to check.
        key: The key to look for.

    Returns:
        True if the key exists, False otherwise.
    """
    return key in data


def check_value_for_key(data: dict, key: str) -> bool:
    """
    Checks if a key exists and its value is not None.

    Using `data.get(key) is not None` is a robust way to check for a non-null value.

    Args:
        data: The dictionary to check.
        key: The key whose value to check.

    Returns:
        True if the key has a non-None value, False otherwise.
    """
    return data.get(key) is not None


def write_json_to_file(data: dict, filename: str) -> None:
    """
    Writes a dictionary to a file as pretty-printed JSON.

    Args:
        data: The dictionary to write.
        filename: The name of the file to create/overwrite.
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4, sort_keys=True)


# ==============================================================================
# SECTION 2: List and Collection Operations
# ==============================================================================

def square_list_items(numbers: list[int | float]) -> list[int | float]:
    """
    Returns a new list where each item is the square of the original.

    Args:
        numbers: A list of numbers.

    Returns:
        A new list with squared numbers.
    """
    return [x * x for x in numbers]


def remove_empty_strings(string_list: list[str]) -> list[str]:
    """
    Removes all empty strings from a list of strings.

    A simple boolean check `if s` is the most Pythonic way to filter empty strings.

    Args:
        string_list: A list that may contain empty strings.

    Returns:
        A new list with empty strings removed.
    """
    return [s for s in string_list if s]


def remove_all_occurrences(sample_list: list, value_to_remove) -> list:
    """
    Removes all occurrences of a specific value from a list.

    Args:
        sample_list: The list to filter.
        value_to_remove: The value to remove from the list.

    Returns:
        A new list without the specified value.
    """
    return [item for item in sample_list if item != value_to_remove]


def count_element_occurrences(sample_list: list) -> dict:
    """
    Counts the occurrences of each element in a list.

    Using `collections.Counter` is the most efficient and standard way to do this.

    Args:
        sample_list: The list of items to count.

    Returns:
        A dictionary mapping each item to its count.
    """
    return dict(Counter(sample_list))


def create_set_from_list_pairs(list1: list, list2: list) -> set:
    """
    Creates a set of tuples by pairing elements from two lists.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        A set of paired tuples.
    """
    return set(zip(list1, list2))


def remove_intersection_from_set(set1: set, set2: set) -> set:
    """
    Removes elements from the first set that are also present in the second set.

    Using the `-` operator or `difference_update` is more idiomatic than iterating.

    Args:
        set1: The set to modify.
        set2: The set containing elements to remove from set1.

    Returns:
        The first set after removing common elements.
    """
    set1.difference_update(set2)
    return set1


def get_unique_dict_values(data: dict) -> list:
    """
    Gets all unique values from a dictionary.

    Converting values to a set is the most direct way to find unique items.

    Args:
        data: The dictionary to extract values from.

    Returns:
        A list of unique values.
    """
    return list(set(data.values()))

def find_largest_item(items: list) -> any:
    """
    Returns the largest item from a list using the built-in max() function.

    Args:
        items: A list of comparable items.

    Returns:
        The largest item in the list.
    """
    if not items:
        return None  # Handle empty list case
    return max(items)

def generate_even_numbers(start: int, end: int) -> list[int]:
    """
    Generates a list of all even numbers within a given range.

    Args:
        start: The starting number of the range (inclusive).
        end: The ending number of the range (exclusive).

    Returns:
        A list of even numbers.
    """
    # Adjust start to the first even number if it's odd
    first_even = start if start % 2 == 0 else start + 1
    return list(range(first_even, end, 2))

# ==============================================================================
# SECTION 3: String Operations
# ==============================================================================

def append_string_in_middle(s1: str, s2: str) -> str:
    """
    Inserts a string (s2) into the middle of another string (s1).

    Args:
        s1: The base string.
        s2: The string to insert.

    Returns:
        The newly formed string.
    """
    middle_index = len(s1) // 2
    return s1[:middle_index] + s2 + s1[middle_index:]


def sort_string_by_case(input_str: str) -> str:
    """
    Arranges string characters so that all lowercase letters come first.

    Args:
        input_str: The string to sort.

    Returns:
        A new string with lowercase letters followed by uppercase letters.
    """
    lower_chars = [char for char in input_str if char.islower()]
    upper_chars = [char for char in input_str if char.isupper()]
    return "".join(lower_chars + upper_chars)


def get_sum_and_avg_of_digits_in_string(input_str: str) -> tuple[int, float]:
    """
    Finds all numbers in a string, and returns their sum and average.

    Args:
        input_str: The string containing numbers and other characters.

    Returns:
        A tuple containing the total sum and the average.
    """
    numbers = [int(num) for num in re.findall(r'\d+', input_str)]
    if not numbers:
        return 0, 0.0
    total = sum(numbers)
    average = total / len(numbers)
    return total, average


def count_character_occurrences(input_str: str) -> dict[str, int]:
    """
    Counts occurrences of all characters within a string.
    
    `collections.Counter` is the ideal tool for this.

    Args:
        input_str: The string to analyze.

    Returns:
        A dictionary mapping each character to its count.
    """
    return dict(Counter(input_str))


def reverse_string(input_str: str) -> str:
    """
    Reverses a given string using slicing.

    Args:
        input_str: The string to reverse.

    Returns:
        The reversed string.
    """
    return input_str[::-1]


def remove_punctuation(input_str: str) -> str:
    """
    Removes all punctuation from a string.

    Args:
        input_str: The string to clean.

    Returns:
        The string without punctuation.
    """
    return input_str.translate(str.maketrans('', '', string.punctuation))


def extract_integers_from_string(input_str: str) -> str:
    """
    Removes all non-digit characters from a string.

    Args:
        input_str: The string to filter.

    Returns:
        A string containing only the digits from the original string.
    """
    return "".join(filter(str.isdigit, input_str))


def replace_punctuation(input_str: str, replacement: str = '#') -> str:
    """
    Replaces each punctuation character in a string with a replacement character.

    Args:
        input_str: The string to process.
        replacement: The character to use for replacement. Defaults to '#'.

    Returns:
        The string with punctuation replaced.
    """
    for punc in string.punctuation:
        input_str = input_str.replace(punc, replacement)
    return input_str

def reverse_digits_of_integer(number: int) -> list[int]:
    """
    Extracts each digit from an integer in reverse order.
    
    Args:
        number: The integer to process.
        
    Returns:
        A list of digits in reverse order.
    """
    if number < 0:
        number = abs(number) # Handle negative numbers gracefully
        
    digits = []
    if number == 0:
        return [0]
        
    while number > 0:
        digits.append(number % 10)
        number //= 10
    return digits


# ==============================================================================
# SECTION 4: Random Data Generation
# ==============================================================================

def generate_random_divisible_integers(count: int, start: int, end: int, divisor: int) -> list[int]:
    """
    Generates a list of random integers between a range that are divisible by a number.

    Args:
        count: The number of random integers to generate.
        start: The start of the range (inclusive).
        end: The end of the range (exclusive).
        divisor: The number the integers must be divisible by.

    Returns:
        A list of random integers.
    """
    return [random.randrange(start, end, divisor) for _ in range(count)]


def get_random_character(input_str: str) -> str:
    """
    Picks a random character from a string.

    Args:
        input_str: The string to choose from.

    Returns:
        A single random character.
    """
    if not input_str:
        return ""
    return random.choice(input_str)


def generate_random_string(length: int) -> str:
    """
    Generates a random string of a given length using ASCII letters.

    Args:
        length: The desired length of the string.

    Returns:
        A random string.
    """
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))


def get_random_date(start_date_str: str, end_date_str: str) -> str:
    """
    Generates a random date string between two given dates.

    Args:
        start_date_str: The start date ('mm/dd/yyyy').
        end_date_str: The end date ('mm/dd/yyyy').

    Returns:
        A random date string in 'mm/dd/yyyy' format.
    """
    start_time = time.mktime(time.strptime(start_date_str, DATE_FORMAT))
    end_time = time.mktime(time.strptime(end_date_str, DATE_FORMAT))
    
    random_time = random.uniform(start_time, end_time)
    
    return time.strftime(DATE_FORMAT, time.localtime(random_time))

def shuffle_list(data: list) -> list:
    """
    Shuffles a list in-place and returns it.

    Args:
        data: The list to be shuffled.

    Returns:
        The shuffled list.
    """
    random.shuffle(data)
    return data

def generate_random_float_array(rows: int, cols: int) -> np.ndarray:
    """
    Generates a random n-dimensional array of floats between [0.0, 1.0).

    Args:
        rows: Number of rows.
        cols: Number of columns.

    Returns:
        A numpy array with random floats.
    """
    return np.random.rand(rows, cols)

def generate_uuid() -> uuid.UUID:
    """
    Generates a random and safe Universally Unique ID (UUID).
    
    Returns:
        A UUID object.
    """
    return uuid.uuid4()

def choose_weighted_random_elements(population: list, weights: list[float], k: int) -> list:
    """
    Chooses k elements from a list with specified weights.

    Args:
        population: The list of elements to choose from.
        weights: The relative weights for each element.
        k: The number of elements to choose.

    Returns:
        A list of k chosen elements.
    """
    return random.choices(population, weights=weights, k=k)

def generate_secure_random_number(upper_bound: int) -> int:
    """
    Generates a cryptographically secure random integer.

    Args:
        upper_bound: The upper bound for the random number (exclusive).
        
    Returns:
        A secure random integer in the range [0, upper_bound-1].
    """
    return secrets.randbelow(upper_bound)
    
# ==============================================================================
# SECTION 5: Date and Time
# ==============================================================================

def string_to_datetime(date_str: str, fmt: str = '%b %d %Y %I:%M%p') -> datetime:
    """
    Converts a string into a datetime object.

    Args:
        date_str: The string representation of the date.
        fmt: The format code for parsing.

    Returns:
        A datetime object.
    """
    return datetime.strptime(date_str, fmt)


def subtract_days_from_date(date_obj: datetime, days: int) -> datetime:
    """
    Subtracts a number of days from a given date.

    Args:
        date_obj: The initial datetime object.
        days: The number of days to subtract.

    Returns:
        The new datetime object.
    """
    return date_obj - timedelta(days=days)


def get_day_of_week(date_obj: datetime) -> str:
    """
    Finds the day of the week for a given date.

    Args:
        date_obj: The datetime object.

    Returns:
        The full name of the day (e.g., 'Sunday').
    """
    return date_obj.strftime('%A')


def add_time_to_date(date_obj: datetime, days: int = 0, hours: int = 0) -> datetime:
    """
    Adds a specified number of days and hours to a date.

    Args:
        date_obj: The initial datetime object.
        days: Number of days to add.
        hours: Number of hours to add.

    Returns:
        The new datetime object.
    """
    return date_obj + timedelta(days=days, hours=hours)


def days_between_dates(date1: datetime, date2: datetime) -> int:
    """
    Calculates the absolute number of days between two dates.
    Using `abs()` simplifies the logic.
    
    Args:
        date1: The first datetime object.
        date2: The second datetime object.
        
    Returns:
        The number of days between the dates.
    """
    delta = abs(date2.date() - date1.date())
    return delta.days


# ==============================================================================
# SECTION 6: Algorithms and Mathematical Functions
# ==============================================================================

def recursive_sum(n: int) -> int:
    """
    Calculates the sum of numbers from n down to 0 using recursion.

    Args:
        n: A non-negative integer.

    Returns:
        The sum of numbers from 0 to n.
    """
    if n <= 0:
        return 0
    return n + recursive_sum(n - 1)


def calculate_median(numbers: list[int | float]) -> int | float:
    """
    Calculates the median of a list of numbers.
    Fixes the logic for even-lengthed lists from the original code.

    Args:
        numbers: A list of numbers.

    Returns:
        The median value.
    """
    if not numbers:
        return None
        
    sorted_nums = sorted(numbers)
    n = len(sorted_nums)
    mid_index = n // 2

    if n % 2 == 1:
        # Odd number of elements
        return sorted_nums[mid_index]
    else:
        # Even number of elements
        return (sorted_nums[mid_index - 1] + sorted_nums[mid_index]) / 2


def calculate_mean(numbers: list[int | float]) -> float:
    """
    Calculates the mean (average) of a list of numbers.
    Using `sum()` and `len()` is more direct.
    
    Args:
        numbers: A list of numbers.
        
    Returns:
        The mean of the numbers.
    """
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)


def normalize_array(arr: list[float]) -> list[float]:
    """
    Normalizes an array of numbers so that they sum to 1.
    
    Args:
        arr: A list of numbers.
        
    Returns:
        A list where elements are normalized.
    """
    total = sum(arr)
    if total == 0:
        return [0.0] * len(arr) # Avoid division by zero
    return [x / total for x in arr]


def softmax(arr: list[float]) -> list[float]:
    """
    Applies the softmax function to an array of numbers.
    
    Args:
        arr: A list of numbers (logits).
    
    Returns:
        A list of probabilities after applying softmax.
    """
    exps = [math.exp(x) for x in arr]
    total = sum(exps)
    if total == 0:
        return [0.0] * len(arr)
    return [ex / total for ex in exps]


def is_armstrong_number(n: int) -> bool:
    """
    Checks if a number is an Armstrong number.
    An Armstrong number is a number that is equal to the sum of its own digits
    each raised to the power of the number of digits.
    
    Args:
        n: The integer to check.
        
    Returns:
        True if it's an Armstrong number, False otherwise.
    """
    if not isinstance(n, int) or n < 0:
        return False
        
    s = str(n)
    power = len(s)
    return n == sum(int(digit)**power for digit in s)


# ==============================================================================
# SECTION 7: Miscellaneous
# ==============================================================================

def get_memory_usage(obj: object) -> int:
    """
    Calculates the memory size of a Python object in bytes.

    Args:
        obj: The Python object.

    Returns:
        The size of the object in bytes.
    """
    return sys.getsizeof(obj)


def are_all_elements_identical(data: list) -> bool:
    """
    Checks if all elements in a list are identical.
    A more efficient way is to convert to a set and check its length.

    Args:
        data: The list to check.

    Returns:
        True if all elements are the same, False otherwise.
    """
    if not data:
        return True  # An empty list has all identical elements vacuously
    return len(set(data)) == 1


def merge_dictionaries(dict1: dict, dict2: dict) -> dict:
    """
    Merges two dictionaries. Values from the second dictionary
    overwrite values from the first in case of key conflicts.

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.

    Returns:
        A new merged dictionary.
    """
    return {**dict1, **dict2}


def lists_to_dictionary(keys: list, values: list) -> dict:
    """
    Converts two lists into a dictionary.

    Args:
        keys: The list to be used as keys.
        values: The list to be used as values.

    Returns:
        A dictionary created from the two lists.
    """
    return dict(zip(keys, values))


def swap_string_case(input_str: str) -> str:
    """
    Converts lowercase letters to uppercase and vice versa in a string.
    The original implementation using ord() and chr() is clever but
    `str.swapcase()` is built-in, more readable, and handles all unicode cases.
    
    Args:
        input_str: The string to modify.
        
    Returns:
        A new string with swapped cases.
    """
    return input_str.swapcase()


# ==============================================================================
# MAIN Function to demonstrate usage
# ==============================================================================

def main():
    """
    Main function to execute and demonstrate all the refactored code snippets.
    """
    print("--- Python Code Examples: Refactored ---\n")

    # --- JSON Operations ---
    print("--- 1. JSON Operations ---")
    person_string = '{"name": "Bob", "languages": ["English", "French"], "numbers": [2, 1.6, null]}'
    person_dict = json.loads(person_string)
    print("Pretty Printed JSON:\n", pretty_print_json(person_dict))
    
    key_to_check = "languages"
    print(f"Does key '{key_to_check}' exist? {check_key_in_json_data(person_dict, key_to_check)}")
    
    key_with_value = "name"
    print(f"Does key '{key_with_value}' have a value? {check_value_for_key(person_dict, key_with_value)}")

    sample_json_data = {"id": 1, "name": "value2", "age": 29}
    output_filename = "sample_output.json"
    write_json_to_file(sample_json_data, output_filename)
    print(f"JSON data written to '{output_filename}'\n")

    # --- List and Collection Operations ---
    print("--- 2. List and Collection Operations ---")
    num_list = [1, 2, 3, 4, 5, 6, 7]
    print(f"Original list: {num_list}, Squared list: {square_list_items(num_list)}")

    str_list = ["Mike", "", "Emma", "Kelly", "", "Brad"]
    print(f"Original list: {str_list}, With empty strings removed: {remove_empty_strings(str_list)}")

    list_to_filter = [5, 20, 15, 20, 25, 50, 20]
    print(f"List {list_to_filter} with all 20s removed: {remove_all_occurrences(list_to_filter, 20)}")
    
    list_to_count = [11, 45, 8, 11, 23, 45, 23, 45, 89]
    print(f"Occurrences in {list_to_count}: {count_element_occurrences(list_to_count)}")

    speed_dict = {'jan': 47, 'feb': 52, 'march': 47, 'April': 44, 'May': 52}
    print(f"Unique values from speed dictionary: {get_unique_dict_values(speed_dict)}\n")

    # --- String Operations ---
    print("--- 3. String Operations ---")
    print(f"Appending 'Kelly' in middle of 'Ault': {append_string_in_middle('Ault', 'Kelly')}")
    print(f"Sorting 'PyNaTive' by case: {sort_string_by_case('PyNaTive')}")
    
    marks_str = "English = 78 Science = 83 Math = 68 History = 65"
    total, avg = get_sum_and_avg_of_digits_in_string(marks_str)
    print(f"For '{marks_str}', Total Marks: {total}, Average: {avg:.2f}")

    print(f"Reversing 'PYnative': {reverse_string('PYnative')}")
    
    punctuation_str = "/*Jon is @developer & musician!!"
    print(f"Removing punctuation from '{punctuation_str}': {remove_punctuation(punctuation_str)}")
    
    integer_str = 'I am 25 years and 10 months old'
    print(f"Extracting integers from '{integer_str}': {extract_integers_from_string(integer_str)}\n")

    # --- Random Data Generation ---
    print("--- 4. Random Data Generation ---")
    print(f"3 random integers between 100-999 divisible by 5: {generate_random_divisible_integers(3, 100, 999, 5)}")
    print(f"Random 5-character string: {generate_random_string(5)}")
    print(f"Random date between 1/1/2016 and 12/12/2018: {get_random_date('1/1/2016', '12/12/2018')}")
    print(f"Shuffled list [1,2,3,4,5]: {shuffle_list([1,2,3,4,5])}")
    print(f"Generated UUID: {generate_uuid()}\n")

    # --- Date and Time ---
    print("--- 5. Date and Time ---")
    dt_obj = string_to_datetime("Feb 25 2020 4:20PM")
    print(f"String to datetime: {dt_obj}")
    print(f"Date a week ago: {subtract_days_from_date(dt_obj, 7)}")
    print(f"Day of the week: {get_day_of_week(dt_obj)}")
    
    date1 = datetime(2020, 2, 25)
    date2 = datetime(2020, 9, 17)
    print(f"Days between {date1.date()} and {date2.date()}: {days_between_dates(date1, date2)}\n")

    # --- Algorithms and Math ---
    print("--- 6. Algorithms and Math ---")
    print(f"Recursive sum of numbers up to 10: {recursive_sum(10)}")
    
    median_list = [2, 5, 1, 8, 4, 9]
    print(f"Median of {median_list}: {calculate_median(median_list)}")
    
    print(f"Is 153 an Armstrong number? {is_armstrong_number(153)}")
    print(f"Is 120 an Armstrong number? {is_armstrong_number(120)}\n")

    # --- Miscellaneous ---
    print("--- 7. Miscellaneous ---")
    list_for_mem = ['Scott', 'Eric', 'Kelly', 'Emma', 'Smith']
    print(f"Memory usage of list {list_for_mem}: {get_memory_usage(list_for_mem)} bytes")
    
    dict1 = {1: 'Scott', 2: "Eric", 3:"Kelly"}
    dict2 = {2: 'Eric', 4: "Emma"}
    print(f"Merged dictionaries: {merge_dictionaries(dict1, dict2)}")
    
    print(f"Swapping case of 'PyThoN ExaMPle': {swap_string_case('PyThoN ExaMPle')}")

if __name__ == "__main__":
    main()