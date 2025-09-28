"""
Refactored Python Code Examples.

This script contains refactored versions of the original code snippets.
The key improvements include:

1.  **Encapsulation**: Each piece of logic is wrapped in a well-named function.
2.  **Readability**:
    -   Variable names are descriptive (e.g., 'c' -> 'ordered_dict').
    -   Modern f-strings are used for printing.
    -   Pythonic idioms like comprehensions and built-in functions (sum, all, any)
        are used to make code more concise and clear.
3.  **Standard Practices**:
    -   Docstrings explain what each function does, its parameters, and what it returns.
    -   Type hints are used for better code clarity and maintainability.
    -   The script is organized with functions first, followed by a main execution
        block (`if __name__ == '__main__':`) to demonstrate usage.
4.  **Efficiency**: Inefficient patterns like nested loops for lookups have been
    replaced with more efficient methods (e.g., using sets for membership testing).
5.  **Correctness**: Bugs or logical errors in the original snippets (e.g., using
    degrees instead of radians for math functions) have been corrected.
"""

import ast
import calendar
import getpass
import itertools
import math
import multiprocessing
import os
import platform
import sys
import textwrap
from collections import Counter
from copy import deepcopy
from datetime import date
from typing import Any, Dict, List, Tuple, Set, Union, Iterable, Generator


# region: Dictionary Examples (Original snippets unnumbered to #100)

def order_dict_by_key_list(data_dict: Dict, key_order: List) -> Dict:
    """Creates a new dictionary ordered according to a list of keys.

    Args:
        data_dict: The original dictionary.
        key_order: A list of keys specifying the desired order.

    Returns:
        A new dictionary with keys sorted as per key_order.
    """
    return {key: data_dict[key] for key in key_order if key in data_dict}


def extract_rows_with_numerical_first_value(data_dict: Dict[Any, List]) -> List[Tuple]:
    """
    Extracts dictionary value lists where the first element across all lists is a digit.

    This function uses a zip(*...) trick to transpose the "rows" and "columns"
    of the dictionary's values. It then checks the first element of each new row.

    Args:
        data_dict: A dictionary where values are lists of the same length.

    Returns:
        A list of tuples, where each tuple is a "row" whose first element was a string digit.
    """
    # zip(*test_dict.values()) transposes the lists:
    # [('34', '875', '98'), ('45', None, 'abc'), ('geeks', '15', '12k')]
    res = []
    # Using try-except to handle cases where an element is not a string (e.g., None)
    for col1, col2, col3 in zip(*data_dict.values()):
        try:
            if col1.isdigit():
                res.append((col1, col2, col3))
        except AttributeError:
            continue  # Ignore non-string types that don't have .isdigit()
    return res


def count_dicts_in_list(data_list: List) -> int:
    """Counts the number of dictionaries within a list.

    Args:
        data_list: A list containing mixed data types.

    Returns:
        The total count of dictionaries in the list.
    """
    return sum(1 for item in data_list if isinstance(item, dict))


def filter_and_double_values(data_dict: Dict[Any, int], k: int) -> Dict[Any, int]:
    """
    Creates a new dictionary, doubling values that are greater than k.

    Args:
        data_dict: A dictionary with integer values.
        k: The threshold value.

    Returns:
        A new dictionary with values > k doubled.
    """
    return {
        key: value * 2 if value > k else value
        for key, value in data_dict.items()
    }


def expand_frequency_dict(freq_dict: Dict[Any, int]) -> List:
    """Converts a frequency dictionary to a list of repeated items.

    Args:
        freq_dict: A dictionary mapping items to their frequency (count).

    Returns:
        A list where each item is repeated according to its frequency.
    """
    return [
        key for key, count in freq_dict.items() for _ in range(count)
    ]


def add_key_value_to_dicts_in_list(
    dict_list: List[Dict], new_key: Any, value_list: List
) -> List[Dict]:
    """
    Assigns a new key and corresponding value to each dictionary in a list.

    This function creates a new list of new dictionaries, preserving the original list.

    Args:
        dict_list: A list of dictionaries.
        new_key: The new key to add to each dictionary.
        value_list: A list of values to assign. Must be the same length as dict_list.

    Returns:
        A new list of updated dictionaries.
    """
    return [
        {**sub_dict, new_key: val}
        for sub_dict, val in zip(dict_list, value_list)
    ]


def check_all_true_in_dict_values(data_dict: Dict[Any, bool]) -> bool:
    """Checks if all boolean values in a dictionary are True.

    Args:
        data_dict: A dictionary with boolean values.

    Returns:
        True if all values are True, False otherwise.
    """
    return all(data_dict.values())


def sum_of_value_lengths(data_dict: Dict[Any, str]) -> int:
    """Calculates the total length of all string values in a dictionary.

    Args:
        data_dict: A dictionary with string values.

    Returns:
        The sum of the lengths of all string values.
    """
    return sum(len(value) for value in data_dict.values())


def get_keys_with_shortest_list(data_dict: Dict[Any, List]) -> List:
    """Finds all keys in a dictionary that correspond to the shortest list(s).

    Args:
        data_dict: A dictionary where values are lists.

    Returns:
        A list of keys that have the shortest lists as values.
    """
    if not data_dict:
        return []
    min_len = min(len(v) for v in data_dict.values())
    return [k for k, v in data_dict.items() if len(v) == min_len]


def decrement_dict_values(data_dict: Dict[Any, int], k: int) -> Dict:
    """Decrements all integer values in a dictionary by a constant K.

    Args:
        data_dict: A dictionary with numerical values.
        k: The number to subtract from each value.

    Returns:
        A new dictionary with the decremented values.
    """
    return {key: value - k for key, value in data_dict.items()}


def count_common_items(dict1: Dict, dict2: Dict) -> int:
    """Counts the number of common key-value pairs between two dictionaries.

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.

    Returns:
        The number of items (key-value pairs) present in both dictionaries.
    """
    # The & operator on dictionary views finds the intersection (common items)
    return len(dict1.items() & dict2.items())


def get_nth_largest_value(data_dict: Dict[Any, int], n: int) -> Union[int, None]:
    """Finds the Nth largest value in a dictionary.

    Args:
        data_dict: A dictionary with numerical values.
        n: The rank of the largest value to find (e.g., n=1 for the largest).

    Returns:
        The Nth largest value, or None if N is out of bounds.
    """
    if not data_dict or n <= 0:
        return None
    # Get unique values, sort them descending, and find the Nth one.
    sorted_values = sorted(list(set(data_dict.values())), reverse=True)
    return sorted_values[n - 1] if n <= len(sorted_values) else None


def get_consecutive_column_diff(tuple_list: List[Tuple], k: int) -> List[int]:
    """
    Calculates the absolute difference between consecutive elements in the Kth column.

    Args:
        tuple_list: A list of tuples.
        k: The index of the column (0-based).

    Returns:
        A list of absolute differences.
    """
    return [
        abs(tup2[k] - tup1[k])
        for tup1, tup2 in zip(tuple_list, tuple_list[1:])
    ]


def filter_positive_tuples(tuple_list: List[Tuple[int, ...]]) -> List[Tuple[int, ...]]:
    """Filters a list of tuples, keeping only those with all non-negative elements.

    Args:
        tuple_list: A list of tuples containing numbers.

    Returns:
        A new list containing only tuples with all non-negative elements.
    """
    return [tup for tup in tuple_list if all(element >= 0 for element in tup)]


def remove_char_from_tuple_strings(
    tuple_list: List[Tuple[str, Any]], char_to_remove: str
) -> List[Tuple[str, Any]]:
    """Removes a specific character from the first element of each tuple in a list.

    Args:
        tuple_list: A list of tuples where the first element is a string.
        char_to_remove: The character to remove.

    Returns:
        A new list of tuples with the character removed from the first element.
    """
    return [
        (sub[0].replace(char_to_remove, ''), sub[1])
        for sub in tuple_list
    ]


def remove_elements_by_type(data_tuple: Tuple, type_to_remove: type) -> Tuple:
    """Creates a new tuple by removing all elements of a specific data type.

    Args:
        data_tuple: The original tuple.
        type_to_remove: The data type to remove (e.g., int, str).

    Returns:
        A new tuple without elements of the specified type.
    """
    return tuple(item for item in data_tuple if not isinstance(item, type_to_remove))


def extract_rear_elements(list_of_tuples: List[Tuple]) -> List:
    """Extracts the last element from each tuple in a list.

    Args:
        list_of_tuples: A list of tuples.

    Returns:
        A list containing the last element of each tuple.
    """
    return [record[-1] for record in list_of_tuples]


def tuple_elementwise_power(tup1: Tuple[int, ...], tup2: Tuple[int, ...]) -> Tuple[int, ...]:
    """
    Computes a new tuple where elements are from tup1 raised to the power of elements from tup2.

    Args:
        tup1: The base tuple.
        tup2: The exponent tuple.

    Returns:
        A new tuple with the results of the element-wise power operation.
    """
    return tuple(base ** exp for base, exp in zip(tup1, tup2))


def count_elements_before_tuple(data_tuple: Tuple) -> int:
    """Counts the number of elements in a tuple before the first nested tuple is found.

    Args:
        data_tuple: The tuple to inspect.

    Returns:
        The count of elements before the first tuple.
    """
    count = 0
    for item in data_tuple:
        if isinstance(item, tuple):
            break
        count += 1
    return count


def get_dissimilar_tuple_elements(tup1: Tuple, tup2: Tuple) -> Tuple:
    """Finds elements that are in one tuple but not both (symmetric difference).

    Args:
        tup1: The first tuple.
        tup2: The second tuple.

    Returns:
        A new tuple containing the dissimilar elements.
    """
    return tuple(set(tup1).symmetric_difference(set(tup2)))


def flatten_tuple_list_to_string(tuple_list: List[Tuple[str, ...]]) -> str:
    """Flattens a list of string tuples into a single space-separated string.

    Args:
        tuple_list: A list of tuples containing strings.

    Returns:
        A single string with all elements joined by spaces.
    """
    return ' '.join(item for tup in tuple_list for item in tup)


def filter_tuples_by_elements(
    tuple_list: List[Tuple], target_list: List
) -> List[Tuple]:
    """
    Filters a list of tuples, keeping only those that contain at least one element
    from a target list.

    Args:
        tuple_list: The list of tuples to filter.
        target_list: The list of elements to search for.

    Returns:
        A filtered list of tuples.
    """
    # Using a set for the target list provides faster lookups
    target_set = set(target_list)
    return [
        tup for tup in tuple_list
        if any(item in target_set for item in tup)
    ]


def concatenate_tuples(tup1: Tuple, tup2: Tuple) -> Tuple:
    """Concatenates two tuples.

    Args:
        tup1: The first tuple.
        tup2: The second tuple.

    Returns:
        A new tuple containing elements of both tuples.
    """
    return tup1 + tup2


def sort_lists_in_tuple(tuple_of_lists: Tuple[List, ...]) -> Tuple[List, ...]:
    """Sorts each individual list contained within a tuple.

    Args:
        tuple_of_lists: A tuple where each element is a list.

    Returns:
        A new tuple with each inner list sorted.
    """
    return tuple(sorted(sub_list) for sub_list in tuple_of_lists)


def remove_strings_from_tuples_in_list(data_list: List[Tuple]) -> List[Tuple]:
    """Removes all string elements from each tuple within a list.

    Args:
        data_list: A list of tuples.

    Returns:
        A new list of tuples with string elements removed.
    """
    return [
        tuple(item for item in tup if not isinstance(item, str))
        for tup in data_list
    ]


def remove_matching_tuples(list1: List[Tuple], list2: List[Tuple]) -> List[Tuple]:
    """Removes tuples from list1 that are also present in list2, preserving order.

    Args:
        list1: The main list of tuples.
        list2: The list of tuples to remove.

    Returns:
        A new list containing tuples from list1 that are not in list2.
    """
    remove_set = set(list2)
    return [item for item in list1 if item not in remove_set]


def group_tuple(data_tuple: Tuple, n: int) -> Tuple[Tuple, ...]:
    """Splits a tuple into a tuple of smaller tuples of size n.

    Args:
        data_tuple: The tuple to split.
        n: The desired size of each chunk.

    Returns:
        A tuple of tuples.
    """
    return tuple(data_tuple[i : i + n] for i in range(0, len(data_tuple), n))


def extract_unique_digits_from_tuples(
    data_list: List[Tuple[int, ...]]
) -> List[str]:
    """
    Extracts all unique digits from numbers inside a list of tuples.

    Args:
        data_list: A list of tuples containing integers.

    Returns:
        A sorted list of unique digits as strings.
    """
    all_digits = {
        digit
        for tup in data_list
        for num in tup
        for digit in str(num)
    }
    return sorted(list(all_digits))


def join_tuple_strings(list_of_tuples: List[Tuple[str, ...]]) -> List[str]:
    """Joins the string elements of each tuple in a list into a single string.

    Args:
        list_of_tuples: A list of tuples, where each tuple contains strings.

    Returns:
        A list of space-joined strings.
    """
    return [' '.join(tup) for tup in list_of_tuples]


def get_max_value_for_keys(dict_list: List[Dict]) -> Dict:
    """
    Finds the maximum value for each key across a list of dictionaries.

    Args:
        dict_list: A list of dictionaries.

    Returns:
        A single dictionary mapping each key to its highest found value.
    """
    max_values: Dict[Any, Any] = {}
    for d in dict_list:
        for key, value in d.items():
            # Update the value if the key is new or the current value is larger
            if key not in max_values or value > max_values[key]:
                max_values[key] = value
    return max_values


def get_keys_by_value_type(data_dict: Dict, target_type: type) -> List:
    """Extracts keys from a dictionary that have values of a specific type.

    Args:
        data_dict: The dictionary to search.
        target_type: The data type to match (e.g., int, str, dict).

    Returns:
        A list of keys whose values match the target type.
    """
    return [key for key, val in data_dict.items() if isinstance(val, target_type)]


# endregion

# region: General Function Examples (Original snippets #1 to #73)

# --- List Functions ---

def remove_last_element(data_list: List) -> List:
    """Removes the last element from a list, modifying it in place.

    Args:
        data_list: The list to modify.

    Returns:
        The modified list.
    """
    if data_list:
        data_list.pop()
    return data_list


def list_is_empty(data_list: List) -> bool:
    """Checks if a list is empty.

    Args:
        data_list: The list to check.

    Returns:
        True if the list is empty, False otherwise.
    """
    return not data_list


def convert_string_to_list(str_list: str) -> List:
    """Safely converts a string representation of a list into a list object.

    Args:
        str_list: The string to convert (e.g., "['a', 1, True]").

    Returns:
        The resulting list.
    """
    return ast.literal_eval(str_list)


def extend_list_without_append(list1: List, list2: List) -> List:
    """Extends a list with another list in-place and returns the modified list.

    Args:
        list1: The list to be extended.
        list2: The list providing the new elements.

    Returns:
        The modified list1.
    """
    list1.extend(list2)
    return list1


def power_list_elements(data_list: List[int], exponent: int) -> List[int]:
    """Returns a new list with each element raised to a given power.

    Args:
        data_list: A list of numbers.
        exponent: The power to raise each number to.

    Returns:
        A new list with the powered elements.
    """
    return [item ** exponent for item in data_list]


# --- Number and Math Functions ---

def find_largest_of_three(num1: int, num2: int) -> int:
    """Finds the largest among two numbers and their sum.

    Args:
        num1: The first number.
        num2: The second number.

    Returns:
        The largest value among num1, num2, and their sum.
    """
    return max(num1, num2, num1 + num2)


def multiply_all_in_list(numbers: List[Union[int, float]]) -> Union[int, float]:
    """Multiplies all the numbers in a list.

    Args:
        numbers: A list of numbers.

    Returns:
        The product of all numbers. Returns 1 for an empty list.
    """
    product = 1
    for num in numbers:
        product *= num
    return product


def find_median_of_three(x: int, y: int, z: int) -> int:
    """Finds the median among three given numbers.

    Args:
        x, y, z: Three numbers.

    Returns:
        The median value.
    """
    return sorted([x, y, z])[1]


def calculate_hypotenuse(side1: float, side2: float) -> float:
    """Calculates the hypotenuse of a right-angled triangle.

    Args:
        side1: Length of the first shorter side.
        side2: Length of the second shorter side.

    Returns:
        The length of the hypotenuse.
    """
    return math.sqrt(side1**2 + side2**2)


# --- String Functions ---

def remove_newline(text: str) -> str:
    """Removes leading/trailing whitespace, including newlines, from a string.

    Args:
        text: The input string.

    Returns:
        The stripped string.
    """
    return text.strip()


def convert_str_to_list(text: str) -> List[str]:
    """Splits a string into a list of words.

    Args:
        text: The string to split.

    Returns:
        A list of words.
    """
    return text.split()


def remove_spaces_from_string(text: str) -> str:
    """Removes all space characters from a string.

    Args:
        text: The input string.

    Returns:
        The string with spaces removed.
    """
    return text.replace(' ', '')


def capitalize_first_and_last(text: str) -> str:
    """Capitalizes the first and last letters of each word in a string.

    Args:
        text: The input string.

    Returns:
        The modified string.
    """
    def capitalize_word(word: str) -> str:
        if len(word) > 1:
            return word[0].upper() + word[1:-1] + word[-1].upper()
        return word.upper()

    return ' '.join(capitalize_word(word) for word in text.split())


def remove_duplicate_words(text: str) -> str:
    """Removes duplicate words from a string, preserving order of first appearance.

    Args:
        text: The input string.

    Returns:
        A string with duplicate words removed.
    """
    seen = set()
    return ' '.join(
        seen.add(word) or word for word in text.split() if word not in seen
    )


def reverse_string_if_multiple_of_four(text: str) -> str:
    """Reverses a string only if its length is a multiple of 4.

    Args:
        text: The input string.

    Returns:
        The reversed string or the original string.
    """
    return text[::-1] if len(text) % 4 == 0 else text


def reverse_words_in_string(text: str) -> str:
    """Reverses the order of words in a string.

    Args:
        text: The input string.

    Returns:
        The string with words reversed.
    """
    return ' '.join(reversed(text.split()))


def count_and_display_vowels(text: str) -> Tuple[int, List[str]]:
    """Counts the vowels in a given text and returns the count and a list of them.

    Args:
        text: The input string.

    Returns:
        A tuple containing the count of vowels and a list of the vowels found.
    """
    vowels = "aeiouAEIOU"
    found_vowels = [letter for letter in text if letter in vowels]
    return len(found_vowels), found_vowels


def find_first_repeated_char(text: str) -> Union[str, None]:
    """Finds the first repeated character in a string.

    Args:
        text: The input string.

    Returns:
        The first character that is repeated, or None if no character repeats.
    """
    seen = set()
    for char in text:
        if char in seen:
            return char
        seen.add(char)
    return None


def find_first_repeated_word(text: str) -> Union[str, None]:
    """Finds the first repeated word in a string.

    Args:
        text: The input string.

    Returns:
        The first word that is repeated, or None if no word repeats.
    """
    seen = set()
    for word in text.split():
        if word in seen:
            return word
        seen.add(word)
    return None

# --- Date, System, and Misc Functions ---

def day_diff(date1: date, date2: date) -> int:
    """Calculates the number of days between two dates.

    Args:
        date1: The first date object.
        date2: The second date object.

    Returns:
        The difference in days.
    """
    return abs((date1 - date2).days)


def check_is_vowel(letter: str) -> bool:
    """Checks if a single character is a vowel.

    Args:
        letter: The character to check.

    Returns:
        True if the letter is a vowel, False otherwise.
    """
    return letter.lower() in "aeiou"


def get_os_info() -> str:
    """Gets the OS name and platform.

    Returns:
        A formatted string with OS information.
    """
    return f'OS Name: {os.name}\nPlatform System: {platform.system()}'


def get_cpu_count() -> int:
    """Returns the number of CPUs on the system.

    Returns:
        The number of CPUs.
    """
    return multiprocessing.cpu_count()


def get_filename_extension(filename: str) -> str:
    """Extracts the extension from a filename.

    Args:
        filename: The name of the file.

    Returns:
        The file extension.
    """
    return filename.split('.')[-1] if '.' in filename else ""


# --- Conversion Functions ---

def temp_converter(value: float, unit: str) -> Union[float, None]:
    """Converts temperature between Celsius and Fahrenheit.

    Args:
        value: The temperature value.
        unit: The unit of the input value ('C' or 'F').

    Returns:
        The converted temperature, or None if the unit is invalid.
    """
    if unit.upper() == 'F':
        return (value - 32) * 5.0 / 9.0
    if unit.upper() == 'C':
        return (9.0 / 5.0) * value + 32
    return None


def height_converter_to_cm(feet: int, inches: int) -> float:
    """Converts height from feet and inches to centimeters.

    Args:
        feet: The height in feet.
        inches: The height in inches.

    Returns:
        The total height in centimeters.
    """
    total_inches = (feet * 12) + inches
    return round(total_inches * 2.54, 1)


def distance_converter(feet: float) -> Dict[str, float]:
    """Converts distance in feet to inches, yards, and miles.

    Args:
        feet: The distance in feet.

    Returns:
        A dictionary with the converted distances.
    """
    return {
        "inches": feet * 12,
        "yards": feet / 3.0,
        "miles": feet / 5280.0,
    }


# --- Geometry and Shape Functions ---

def area_of_regular_polygon(num_sides: int, circumradius: float) -> float:
    """Calculates the area of a regular polygon.

    Args:
        num_sides: The number of sides of the polygon.
        circumradius: The radius of the circumscribed circle.

    Returns:
        The area of the polygon.
    """
    return (1/2) * num_sides * (circumradius ** 2) * math.sin(2 * math.pi / num_sides)


def volume_sphere(radius: float) -> float:
    """Calculates the volume of a sphere."""
    return (4/3) * math.pi * (radius ** 3)


def volume_cube(side: float) -> float:
    """Calculates the volume of a cube."""
    return side ** 3


def volume_cylinder(radius: float, height: float) -> float:
    """Calculates the volume of a cylinder."""
    return math.pi * (radius ** 2) * height


# --- Generator Functions ---

def lucas_numbers(n: int) -> Generator[int, None, None]:
    """Generates the first n Lucas Numbers (2, 1, 3, 4, ...).

    Args:
        n: The number of Lucas numbers to generate.
    """
    a, b = 2, 1
    for _ in range(n):
        yield a
        a, b = b, a + b

# endregion


def main():
    """
    Main function to demonstrate the refactored code snippets.
    """
    print("--- Refactored Code Demonstrations ---")

    # Demo for: order_dict_by_key_list
    print("\n1. Order dictionary by a list of keys:")
    test_dict_1 = {'is': 2, 'for': 4, 'gfg': 1, 'best': 3, 'geeks': 5}
    ord_list_1 = ['gfg', 'is', 'best', 'for', 'geeks']
    print(f"Original Dict: {test_dict_1}")
    print(f"Key Order: {ord_list_1}")
    print(f"Ordered Dict: {order_dict_by_key_list(test_dict_1, ord_list_1)}")

    # Demo for: count_dicts_in_list (#71)
    print("\n2. Count dictionaries in a list:")
    test_list_2 = [10, {'gfg': 1}, {'ide': 2, 'code': 3}, 20]
    print(f"Original List: {test_list_2}")
    print(f"Number of Dictionaries: {count_dicts_in_list(test_list_2)}")

    # Demo for: expand_frequency_dict (#73)
    print("\n3. Convert frequency dictionary to list:")
    test_dict_3 = {'gfg': 4, 'is': 2, 'best': 5}
    print(f"Original Dict: {test_dict_3}")
    print(f"Expanded List: {expand_frequency_dict(test_dict_3)}")

    # Demo for: count_common_items (#79)
    print("\n4. Find common items among dictionaries:")
    test_dict_4a = {'gfg': 1, 'is': 2, 'best': 3}
    test_dict_4b = {'gfg': 1, 'is': 2, 'good': 3}
    print(f"Dict 1: {test_dict_4a}")
    print(f"Dict 2: {test_dict_4b}")
    print(f"Number of Common Items: {count_common_items(test_dict_4a, test_dict_4b)}")

    # Demo for: get_nth_largest_value (#80)
    print("\n5. Get Nth largest value in a dictionary:")
    test_dict_5 = {'a': 10, 'b': 30, 'c': 20, 'd': 30, 'e': 5}
    n = 2
    print(f"Dict: {test_dict_5}")
    print(f"The {n}nd largest value is: {get_nth_largest_value(test_dict_5, n)}")
    
    # Demo for: get_consecutive_column_diff (#81)
    print("\n6. Get consecutive Kth column difference in tuple list:")
    test_list_6 = [(5, 4, 2), (1, 3, 4), (5, 7, 8), (7, 4, 3)]
    k = 1
    print(f"List: {test_list_6}, K={k}")
    print(f"Differences: {get_consecutive_column_diff(test_list_6, k)}")
    
    # Demo for: remove_matching_tuples (#94)
    print("\n7. Remove matching tuples:")
    list1_7 = [('Early', 'morning'), ('is', 'good'), ('for', 'Health')]
    list2_7 = [('Early', 'morning'), ('is', 'good')]
    print(f"List 1: {list1_7}")
    print(f"List 2: {list2_7}")
    print(f"Result: {remove_matching_tuples(list1_7, list2_7)}")
    
    # Demo for: find_largest_of_three (Original #2)
    print("\n8. Find the largest of two numbers and their sum:")
    num1, num2 = 10, 25
    print(f"Numbers: {num1}, {num2}. Sum: {num1 + num2}")
    print(f"Largest: {find_largest_of_three(num1, num2)}")

    # Demo for: capitalize_first_and_last (Original #18)
    print("\n9. Capitalize first and last letters of words:")
    text_9 = "python is a fun language"
    print(f"Original: '{text_9}'")
    print(f"Modified: '{capitalize_first_and_last(text_9)}'")

    # Demo for: find_first_repeated_char (Original #36)
    print("\n10. Find the first repeated character:")
    text_10 = "programming"
    print(f"String: '{text_10}'")
    print(f"First repeated char: {find_first_repeated_char(text_10)}")

    # Demo for: lucas_numbers (Original #57)
    print("\n11. Generate Lucas Numbers:")
    print(f"First 10 Lucas numbers: {list(lucas_numbers(10))}")


if __name__ == '__main__':
    main()