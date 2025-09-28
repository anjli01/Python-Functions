# -*- coding: utf-8 -*-
"""
This script contains a collection of refactored Python code examples.

Each example from the original source has been encapsulated into a well-documented
function, following modern Python best practices including PEP 8 styling,
type hinting, and descriptive naming.

The main execution block (`if __name__ == "__main__":`) demonstrates the usage
of each function.
"""

# --- Imports ---
import datetime
import functools
import os
import re
from time import localtime
from typing import Any, Callable, Dict, List, Set, Tuple, Union, Sequence


# --- List and Tuple Operations ---

def check_name_exists(name_to_find: str, names_list: List[str]) -> bool:
    """
    Checks if a name exists in a list, case-insensitively.

    Args:
        name_to_find: The name to search for.
        names_list: The list of names to search within.

    Returns:
        True if the name is found, False otherwise.
    """
    return name_to_find.lower() in [name.lower() for name in names_list]

def get_matrix_column(matrix: List[List[Any]], column_index: int) -> List[Any]:
    """
    Extracts a specific column from a 2D list (matrix).

    Args:
        matrix: The matrix (list of lists).
        column_index: The index of the column to extract.

    Returns:
        A list containing the elements of the specified column.
    """
    if not matrix or not matrix[0]:
        return []
    return [row[column_index] for row in matrix]

def get_alternate_elements(sequence: Sequence) -> List[Any]:
    """
    Returns a list of alternate elements from a sequence (e.g., tuple or list).

    Args:
        sequence: The input sequence.

    Returns:
        A new list containing every other element, starting from the first.
    """
    return list(sequence[::2])

def sort_tuple(input_tuple: Tuple) -> Tuple:
    """
    Sorts the elements of a tuple.

    Args:
        input_tuple: The tuple to sort.

    Returns:
        A new tuple with the elements sorted in ascending order.
    """
    return tuple(sorted(input_tuple))

def multiply_list_elements_cartesian(list1: List[float], list2: List[float]) -> List[float]:
    """
    Calculates the Cartesian product of two lists and multiplies the pairs.

    Args:
        list1: The first list of numbers.
        list2: The second list of numbers.

    Returns:
        A list of all possible products (e.g., [x1*y1, x1*y2, x2*y1, ...]).
    """
    return [x * y for x in list1 for y in list2]

def filter_negative_integers(numbers: List[int]) -> List[int]:
    """
    Filters a list to keep only negative integers.

    Args:
        numbers: A list of integers.

    Returns:
        A new list containing only the negative integers from the input.
    """
    return [num for num in numbers if num < 0]

def add_lists_elementwise(list1: List[float], list2: List[float]) -> List[float]:
    """
    Adds two lists together element by element.

    Args:
        list1: The first list of numbers.
        list2: The second list of numbers.

    Returns:
        A new list where each element is the sum of elements at the same index.
    """
    return [x + y for x, y in zip(list1, list2)]

def transpose_matrix(matrix: List[List[Any]]) -> List[List[Any]]:
    """
    Transposes a matrix (converts rows to columns and vice-versa).

    Args:
        matrix: The matrix to transpose.

    Returns:
        The transposed matrix.
    """
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]

def extend_and_append_to_list(original_list: List[Any]) -> List[Any]:
    """
    Demonstrates the difference between append and extend on a list.

    Args:
        original_list: The list to modify.

    Returns:
        The modified list after appending and extending.
    """
    list_copy = original_list[:]
    list_copy.append([87])  # Appends the list [87] as a single element
    list_copy.extend([45, 67])  # Extends the list with elements 45 and 67
    return list_copy

def get_non_vowels(input_string: str) -> List[str]:
    """
    Extracts all non-vowel characters from a string.

    Args:
        input_string: The string to process.

    Returns:
        A list of characters that are not vowels.
    """
    VOWELS = "aeiou"
    return [char for char in input_string.lower() if char not in VOWELS]

def get_filtered_strings(items: List[str]) -> List[str]:
    """
    Filters a list of strings, keeping only those that are purely alphabetic or numeric.

    Args:
        items: A list of strings.

    Returns:
        A filtered list of strings.
    """
    return [item for item in items if item.isalpha() or item.isdigit()]


# --- Dictionary Operations ---

def get_current_activity(schedule: Dict[int, str]) -> str:
    """
    Determines the current activity based on the hour of the day and a schedule.

    Args:
        schedule: A dictionary mapping an hour (key) to an activity (value).

    Returns:
        A string describing the next scheduled activity or a default message.
    """
    time_now = localtime()
    current_hour = time_now.tm_hour

    for activity_time in sorted(schedule.keys()):
        if current_hour < activity_time:
            return schedule[activity_time]
    
    # This 'else' belongs to the 'for' loop. It runs if the loop completes without a 'break'.
    return 'Unknown, AFK or sleeping!'

def invert_dictionary(input_dict: Dict) -> Dict:
    """
    Inverts a dictionary, swapping keys and values.

    Note: This assumes values are unique and hashable. If values are not unique,
    some key-value pairs will be lost.

    Args:
        input_dict: The dictionary to invert.

    Returns:
        A new dictionary with keys and values swapped.
    """
    return {value: key for key, value in input_dict.items()}

def remove_dict_element(input_dict: Dict, key_to_remove: Any) -> Dict:
    """
    Removes an element from a dictionary by its key using pop.

    Args:
        input_dict: The dictionary to modify.
        key_to_remove: The key of the element to remove.

    Returns:
        The dictionary after the element has been removed.
    """
    input_dict.pop(key_to_remove, None)  # Use None to avoid KeyError if key not found
    return input_dict

def merge_dictionaries(dict1: Dict, dict2: Dict) -> Dict:
    """
    Merges the second dictionary into the first one.

    Args:
        dict1: The primary dictionary (will be modified).
        dict2: The dictionary to merge into the first.

    Returns:
        The modified first dictionary.
    """
    dict1.update(dict2)
    return dict1


# --- String Operations ---

def count_word_in_file(filepath: str, word_to_find: str) -> int:
    """
    Counts the occurrences of a specific word in a text file.

    Args:
        filepath: The path to the text file.
        word_to_find: The word to count.

    Returns:
        The total count of the word.
    """
    count = 0
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.split()
                for word in words:
                    if word == word_to_find:
                        count += 1
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return 0
    return count

def get_string_ascii_values(input_string: str) -> List[int]:
    """
    Converts a string into a list of its character's ASCII values.

    Args:
        input_string: The string to convert.

    Returns:
        A list of integers representing the ASCII values.
    """
    return [ord(char) for char in input_string]

def replace_first_char(word: str, new_char: str) -> str:
    """
    Replaces the first character of a word.

    Args:
        word: The original word.
        new_char: The character to replace the first one with.

    Returns:
        The modified word.
    """
    if not word or not new_char:
        return word
    return new_char + word[1:]


# --- Set Operations ---

def create_frozen_set(items: List[Any]) -> frozenset:
    """
    Creates a frozenset from a list of items.

    Args:
        items: A list of items.

    Returns:
        A frozenset containing unique items from the list.
    """
    return frozenset(items)

def sum_unique_elements(numbers: Set[Union[int, float]], start_value: int = 0) -> Union[int, float]:
    """
    Calculates the sum of elements in a set, with an optional starting value.

    Args:
        numbers: A set of numbers.
        start_value: The value to start the sum from.

    Returns:
        The total sum.
    """
    return sum(numbers, start_value)


# --- Functions, Lambdas, and Generators ---

def demonstrate_unpacking(a: Any, b: Any, c: Any, d: Any) -> Any:
    """
    A simple function to demonstrate argument unpacking. It returns the sum of the
    first and fourth arguments.

    Args:
        a, b, c, d: The four arguments.

    Returns:
        The sum of the first and fourth arguments.
    """
    return a + d

def simple_generator(x: int):
    """
    A simple generator that yields a value, prints a message, and yields another.

    Args:
        x: An integer.
    """
    print("Generator started...")
    yield x + 1
    print("...resuming generator...")
    yield x + 2
    print("...generator finished.")

def make_incrementor(n: int) -> Callable[[int], int]:
    """
    A function factory that returns a lambda function to increment a number.

    Args:
        n: The amount to increment by.

    Returns:
        A new function that takes a number `x` and returns `x + n`.
    """
    return lambda x: x + n


# --- Numeric & Mathematical Operations ---

def compare_numbers(x: Union[int, float], y: Union[int, float]) -> str:
    """
    Compares two numbers and returns a descriptive string.

    Args:
        x: The first number.
        y: The second number.

    Returns:
        A string indicating if x > y, x < y, or x == y.
    """
    if x > y:
        return f"{x} is greater than {y}"
    if x < y:
        return f"{y} is greater than {x}"
    return "The numbers are equal"

def multiply_three_numbers(num1: float, num2: float, num3: float) -> float:
    """Multiplies three numbers together."""
    return num1 * num2 * num3

def get_quotient_and_remainder(num1: int, num2: int) -> Tuple[int, int]:
    """
    Divides the first number by the second.

    Args:
        num1: The dividend.
        num2: The divisor.

    Returns:
        A tuple containing the quotient and the remainder.
    """
    if num2 == 0:
        raise ValueError("Divisor cannot be zero.")
    return divmod(num1, num2)

def calculate_factorial(n: int) -> int:
    """
    Calculates the factorial of a number using functools.reduce.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial of n.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0:
        return 1
    return functools.reduce(lambda x, y: x * y, range(1, n + 1))


# --- Main Execution Block ---

if __name__ == "__main__":
    print("--- Running Python Code Examples ---")

    # --- List and Tuple Operations ---
    print("\n--- Checking if a name exists in a list ---")
    names1 = ['Amir', 'Bala', 'Chales']
    exists = check_name_exists('amir', names1)
    print(f"Does 'amir' exist in {names1}? {'Yes' if exists else 'No'}")
    
    print("\n--- Extracting a column from a matrix ---")
    matrix = [[1, 2, 3, 4], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
    column = get_matrix_column(matrix, 1)
    print(f"Column 1 of the matrix: {column}")

    print("\n--- Getting alternate elements from a tuple ---")
    t = (1, 2, 4, 3, 8, 9)
    alternates = get_alternate_elements(t)
    print(f"Alternate elements of {t}: {alternates}")

    print("\n--- Sorting a tuple ---")
    a = (2, 3, 1, 5)
    sorted_t = sort_tuple(a)
    print(f"Original tuple: {a}, Sorted tuple: {sorted_t}")

    print("\n--- Cartesian product multiplication of two lists ---")
    l1 = [1, 2, 3]
    l2 = [4, 5, 6]
    products = multiply_list_elements_cartesian(l1, l2)
    print(f"Products of {l1} and {l2}: {products}")
    
    print("\n--- Filtering negative integers ---")
    l3 = [1, 2, 3, -4, -8]
    negatives = filter_negative_integers(l3)
    print(f"Negative integers in {l3}: {negatives}")

    print("\n--- Adding two lists element-wise ---")
    l4, l5 = [10, 20, 30], [-10, -20, -30]
    sums = add_lists_elementwise(l4, l5)
    print(f"Sum of {l4} and {l5}: {sums}")

    print("\n--- Transposing a matrix ---")
    matrix_to_transpose = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    transposed = transpose_matrix(matrix_to_transpose)
    print(f"Original matrix: {matrix_to_transpose}, Transposed: {transposed}")
    
    print("\n--- List append vs. extend ---")
    base_list = [13, 56, 17]
    modified = extend_and_append_to_list(base_list)
    print(f"Original: {base_list}, After append/extend: {modified}")
    
    print("\n--- Filtering non-vowels from a string ---")
    non_vowels = get_non_vowels('balaji')
    print(f"Non-vowels in 'balaji': {non_vowels}")
    
    print("\n--- Filtering a list for alphabetic or numeric strings ---")
    str_list = ["good", "oh!", "excellent!", "#450", "123"]
    filtered = get_filtered_strings(str_list)
    print(f"Filtered list from {str_list}: {filtered}")
    
    print("\n--- Unpacking a list into function arguments ---")
    x = [1, 2, 3, 4]
    result_unpack = demonstrate_unpacking(*x)
    print(f"Unpacking {x} into a function that sums first and last elements: {result_unpack}")


    # --- Dictionary Operations ---
    print("\n--- Getting the current activity from a schedule ---")
    activity_schedule = {8: 'Sleeping', 9: 'Commuting', 17: 'Working', 18: 'Commuting', 20: 'Eating', 22: 'Resting'}
    current_activity = get_current_activity(activity_schedule)
    print(f"Current activity: {current_activity}")

    print("\n--- Inverting a dictionary ---")
    dict_to_invert = {"a": 1, "b": 2, "c": 3}
    inverted = invert_dictionary(dict_to_invert)
    print(f"Original dict: {dict_to_invert}, Inverted: {inverted}")
    
    print("\n--- Removing a dictionary element ---")
    my_dict = {1: 5, 2: 3, 3: 4}
    print(f"Original dict: {my_dict}, after removing key 3: {remove_dict_element(my_dict, 3)}")
    
    print("\n--- Merging dictionaries ---")
    d1 = {"john": 40, "peter": 45}
    d2 = {"susan": 50}
    print(f"Merging {d1} and {d2}: {merge_dictionaries(d1, d2)}")

    # --- String Operations ---
    print("\n--- Counting a word in a file ---")
    # Create a dummy file for the example
    with open("sample.txt", "w") as f:
        f.write("this is a sample file with a sample keyword\n")
        f.write("the keyword appears twice in this sample file")
    keyword_count = count_word_in_file('sample.txt', 'sample')
    print(f"The word 'sample' appears {keyword_count} times in sample.txt")
    os.remove("sample.txt") # Clean up the dummy file

    print("\n--- Using various string methods ---")
    print(f"'abcdefcdghcd'.split('cd', 2) -> {'abcdefcdghcd'.split('cd', 2)}")
    print(f"'ab cd-ef'.title() -> {'ab cd-ef'.title()}")
    print(f"'ab'.zfill(5) -> {'ab'.zfill(5)}")
    print(f"'abcdef12'.replace('cd', '12') -> {'abcdef12'.replace('cd', '12')}")
    print(f"'for'.isidentifier() -> {'for'.isidentifier()}")
    print(f"'11'.isnumeric() -> {'11'.isnumeric()}")
    print(f"'ab'.isalpha() -> {'ab'.isalpha()}")
    
    print("\n--- f-string formatting ---")
    var1 = 'Python'
    print(f"f-string is a good feature in {var1}.")

    print("\n--- Replacing first character of a word ---")
    print(f"Replacing first char of 'goal' with 'f': {replace_first_char('goal', 'f')}")
    
    # --- Set Operations ---
    print("\n--- Creating a frozenset ---")
    my_frozenset = create_frozen_set([5, 6, 7, 6, 5])
    print(f"Frozenset from [5, 6, 7, 6, 5]: {my_frozenset}")
    
    print("\n--- Summing unique elements in a set ---")
    my_set = {5, 6, 7}
    print(f"Sum of {my_set} with start value 5: {sum_unique_elements(my_set, 5)}")

    # --- Functions, Lambdas, and Generators ---
    print("\n--- Using a lambda function for cubing ---")
    cube = lambda x: x ** 3
    print(f"Cube of 5 is: {cube(5)}")
    
    print("\n--- Demonstrating a simple generator ---")
    gen = simple_generator(9)
    print(f"First yield: {next(gen)}")
    print(f"Second yield: {next(gen)}")
    
    print("\n--- Using a function factory (make_incrementor) ---")
    add_42 = make_incrementor(42)
    print(f"Using incrementor created with 42 on input 1: {add_42(1)}")

    # --- Numeric & Mathematical Operations ---
    print("\n--- Comparing two numbers ---")
    print(f"Comparing 2 and 3: {compare_numbers(2, 3)}")
    
    print("\n--- Getting quotient and remainder ---")
    q, r = get_quotient_and_remainder(10, 3)
    print(f"10 divided by 3 is quotient {q} and remainder {r}")
    
    print("\n--- Calculating factorial ---")
    print(f"Factorial of 5 is: {calculate_factorial(5)}")
    
    # --- Control Flow & Error Handling ---
    print("\n--- Demonstrating a try-except block ---")
    try:
        # Sets do not support multiplication
        unsupported_op = {5, 6} * 3
    except Exception as e:
        print(f"Caught expected error: {e}")
        
    print("\n--- Demonstrating an assert statement ---")
    x, y = 1, 8
    try:
        assert x > y, 'x is not greater than y'
    except AssertionError as e:
        print(f"Caught expected assertion error: {e}")
        
    print("\n--- Today's date ---")
    print(f"Today is: {datetime.date.today()}")
