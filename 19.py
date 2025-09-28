"""
This module contains a collection of refactored Python code examples.

Each example is encapsulated in a well-documented function, follows standard
Python conventions (PEP 8), and includes type hints for clarity.
The examples cover a wide range of basic operations on dictionaries, lists,
tuples, and more.
"""
import datetime
import math
import time
from collections import Counter, defaultdict
from typing import (Any, Callable, Dict, List, Sequence, Tuple, Union,
                    Optional)

# ==============================================================================
# Dictionary Operations
# ==============================================================================


def combine_dicts_with_priority(
    low_prio_dict: Dict[Any, Any], high_prio_dict: Dict[Any, Any]
) -> Dict[Any, Any]:
    """
    Combines two dictionaries, with one having priority.

    If a key exists in both dictionaries, the value from the high-priority
    dictionary will be used.

    Args:
        low_prio_dict: The dictionary with lower priority.
        high_prio_dict: The dictionary with higher priority.

    Returns:
        A new dictionary representing the merged result.
    """
    # Start with a copy of the low-priority dict to avoid modifying it
    combined = low_prio_dict.copy()
    # Update it with the high-priority dict. Existing keys will be overwritten.
    combined.update(high_prio_dict)
    return combined


def sum_common_key_values(
    dict1: Dict[Any, int], dict2: Dict[Any, int]
) -> Dict[Any, int]:
    """
    Combines two dictionaries by adding values for common keys.

    Uses collections.Counter for a concise and efficient solution.

    Args:
        dict1: The first dictionary with numeric values.
        dict2: The second dictionary with numeric values.

    Returns:
        A new dictionary with combined keys and summed values for common keys.
    """
    return dict(Counter(dict1) + Counter(dict2))


def sort_dict_keys_by_value(data: Dict[Any, Any]) -> List[Any]:
    """
    Sorts dictionary keys into a list based on their corresponding values.

    Args:
        data: The input dictionary.

    Returns:
        A list of keys, sorted in ascending order of their values.
    """
    # The `key` argument of `sorted` specifies a function to be called on each
    # list element prior to making comparisons. `data.get` is a perfect fit.
    return sorted(data, key=data.get)


def concatenate_dict_list_values(
    list_of_dicts: List[Dict[str, List[Any]]]
) -> Dict[str, List[Any]]:
    """
    Concatenates values from a list of dictionaries that share the same key.

    Args:
        list_of_dicts: A list where each item is a dictionary with
                       list values.

    Returns:
        A single dictionary with concatenated list values for each key.
    """
    # defaultdict simplifies the logic by providing a default value (an empty list)
    # for keys that haven't been seen yet.
    result = defaultdict(list)
    for d in list_of_dicts:
        for key, value in d.items():
            result[key].extend(value)
    return dict(result)


def get_top_n_largest_keys(data: Dict[int, Any], n: int) -> List[int]:
    """
    Finds the top N largest keys in a dictionary with integer keys.

    Args:
        data: The input dictionary with integer keys.
        n: The number of largest keys to retrieve.

    Returns:
        A list containing the N largest keys, sorted in descending order.
    """
    # Sort the keys directly and take the first N elements.
    return sorted(data.keys(), reverse=True)[:n]


def extract_value_by_key(
    list_of_dicts: List[Dict[str, Any]], key: str
) -> Optional[Any]:
    """
    Extracts the first value found for a given key from a list of dictionaries.

    Args:
        list_of_dicts: The list of dictionaries to search through.
        key: The key whose value needs to be extracted.

    Returns:
        The value of the first matching key, or None if the key is not found.
    """
    for d in list_of_dicts:
        if key in d:
            return d[key]
    return None

# ==============================================================================
# Date and Time Operations
# ==============================================================================


def date_str_to_timestamp(date_string: str, format_str: str = "%d/%m/%Y") -> float:
    """
    Converts a date string to a Unix timestamp.

    Args:
        date_string: The date string to convert (e.g., "20/01/2020").
        format_str: The format code for parsing the date string.

    Returns:
        The Unix timestamp as a float.
    """
    date_element = datetime.datetime.strptime(date_string, format_str)
    return date_element.timestamp()


def time_12hr_to_24hr(time_12hr: str) -> str:
    """
    Converts a time string from 12-hour format to 24-hour format.

    Args:
        time_12hr: The time string in 12-hour format (e.g., "07:05:45 PM").

    Returns:
        The time string in 24-hour format (e.g., "19:05:45").
    """
    # Using strptime and strftime is the standard, robust way to handle time conversions.
    time_obj = datetime.datetime.strptime(time_12hr, "%I:%M:%S %p")
    return time_obj.strftime("%H:%M:%S")

# ==============================================================================
# List and Sequence Operations
# ==============================================================================


def product_of_integers_in_list(mixed_list: List[Any]) -> int:
    """
    Calculates the product of all integer types in a list with mixed data types.

    Args:
        mixed_list: A list containing various data types.

    Returns:
        The product of all integers found in the list. Returns 1 if no integers.
    """
    product = 1
    for item in mixed_list:
        if isinstance(item, int):
            product *= item
    return product


def add_two_lists_element_wise(list1: List[float], list2: List[float]) -> List[float]:
    """
    Adds two lists element-wise.

    The resulting list will have the length of the shorter input list.

    Args:
        list1: The first list of numbers.
        list2: The second list of numbers.

    Returns:
        A new list containing the element-wise sum.
    """
    return [x + y for x, y in zip(list1, list2)]


def get_positive_numbers(numbers: List[float]) -> List[float]:
    """Filters a list to return only the positive numbers."""
    return [num for num in numbers if num >= 0]


def get_negative_numbers(numbers: List[float]) -> List[float]:
    """Filters a list to return only the negative numbers."""
    return [num for num in numbers if num < 0]


def get_odd_numbers(numbers: List[int]) -> List[int]:
    """Filters a list to return only the odd numbers."""
    return [num for num in numbers if num % 2 != 0]


def remove_even_numbers(numbers: List[int]) -> List[int]:
    """
    Creates a new list containing only the odd numbers from the original list.

    Note: It is unsafe to modify a list while iterating over it. This function
    correctly creates a new list.

    Args:
        numbers: A list of integers.

    Returns:
        A new list with all even numbers removed.
    """
    return get_odd_numbers(numbers)


def multiply_list_elements(numbers: List[float]) -> float:
    """
    Calculates the product of all numbers in a list.

    Args:
        numbers: A list of numbers.

    Returns:
        The product of the numbers. Returns 1 for an empty list.
    """
    if not numbers:
        return 1.0
    # math.prod is available in Python 3.8+ and is highly efficient.
    # For older versions, a loop is fine.
    try:
        return math.prod(numbers)
    except AttributeError:
        result = 1.0
        for x in numbers:
            result *= x
        return result


def remove_unwanted_elements(
    data: List[Any], elements_to_remove: set
) -> List[Any]:
    """Removes a set of specified elements from a list."""
    return [item for item in data if item not in elements_to_remove]


def filter_empty_strings(data: List[str]) -> List[str]:
    """Removes empty or whitespace-only strings from a list of strings."""
    return [s for s in data if s and not s.isspace()]


def get_cumulative_sum(numbers: List[float]) -> List[float]:
    """
    Calculates the cumulative sum of a list of numbers.

    Args:
        numbers: A list of numbers.

    Returns:
        A list where each element is the sum of the numbers up to that index.
    """
    cumulative_list = []
    current_sum = 0
    for num in numbers:
        current_sum += num
        cumulative_list.append(current_sum)
    return cumulative_list


def find_even_indices_distance(numbers: List[int]) -> Optional[int]:
    """
    Calculates the distance between the first and last occurrence of an even number.

    Args:
        numbers: A list of integers.

    Returns:
        The distance (difference in indices), or None if fewer than two
        even numbers are found.
    """
    even_indices = [i for i, num in enumerate(numbers) if num % 2 == 0]
    if len(even_indices) < 2:
        return None
    return even_indices[-1] - even_indices[0]


def reverse_list(data: List[Any]) -> List[Any]:
    """Returns a reversed copy of a list."""
    return data[::-1]


def count_occurrences(data: List[Any], element_to_count: Any) -> int:
    """Counts the number of times an element appears in a list."""
    # The list.count() method is the most direct way to do this.
    return data.count(element_to_count)


def swap_first_and_last(data: List[Any]) -> List[Any]:
    """
    Swaps the first and last elements of a list.

    Args:
        data: The list to be modified.

    Returns:
        The list with elements swapped. Returns the original list if it has
        fewer than two elements.
    """
    if len(data) < 2:
        return data
    # Python's tuple packing/unpacking makes swapping elegant
    data[0], data[-1] = data[-1], data[0]
    return data


def list_of_numbers_to_split_key_value_dict(numbers: List[int]) -> Dict[int, int]:
    """
    Converts a list of integers into a dictionary by splitting each number.

    Each number is converted to a string, split in the middle, and the two
    halves are used as a key-value pair.

    Args:
        numbers: A list of integers.

    Returns:
        A dictionary constructed from the split numbers.
    """
    result_dict = {}
    for num in numbers:
        num_str = str(num)
        mid_index = len(num_str) // 2
        # Handle cases where a number might result in an empty key or value
        if mid_index == 0:
            continue
        key = int(num_str[:mid_index])
        value = int(num_str[mid_index:])
        result_dict[key] = value
    return result_dict


def filter_numbers_with_digit(numbers: List[int], digit: int) -> List[int]:
    """
    Filters a list of numbers to find those containing a specific digit.

    Args:
        numbers: A list of integers.
        digit: The digit to search for (0-9).

    Returns:
        A list of numbers that contain the specified digit.
    """
    digit_str = str(digit)
    return [num for num in numbers if digit_str in str(num)]


def count_unique_elements(data: List[Any]) -> int:
    """Counts the number of unique elements in a list."""
    # Converting to a set is the most Pythonic way to find unique elements.
    return len(set(data))


def get_sum_and_average(numbers: List[float]) -> Tuple[float, float]:
    """
    Calculates the sum and average of a list of numbers.

    Args:
        numbers: A list of numbers.

    Returns:
        A tuple containing the sum and the average. Returns (0, 0) for an
        empty list.
    """
    if not numbers:
        return 0, 0
    total_sum = sum(numbers)
    average = total_sum / len(numbers)
    return total_sum, average

# ==============================================================================
# Tuple Operations
# ==============================================================================


def remove_tuples_of_length(
    list_of_tuples: List[Tuple], length_to_remove: int
) -> List[Tuple]:
    """Filters a list of tuples, removing those of a specific length."""
    return [t for t in list_of_tuples if len(t) != length_to_remove]


def create_number_and_cube_tuples(numbers: List[float]) -> List[Tuple[float, float]]:
    """Creates a list of tuples, where each tuple is a number and its cube."""
    return [(num, num ** 3) for num in numbers]


def swap_tuple_elements(list_of_tuples: List[Tuple]) -> List[Tuple]:
    """Swaps the first two elements of each tuple in a list."""
    return [(y, x) for x, y in list_of_tuples]


def sort_tuples_by_second_item(
    list_of_tuples: List[Tuple],
) -> List[Tuple]:
    """Sorts a list of tuples based on the value of the second item in each tuple."""
    return sorted(list_of_tuples, key=lambda item: item[1])


def all_pair_combinations(tuple1: Tuple, tuple2: Tuple) -> List[Tuple]:
    """Generates all pair combinations from two tuples."""
    # itertools.product is the standard tool for this, but a comprehension is fine.
    return [(a, b) for a in tuple1 for b in tuple2] + [
        (b, a) for b in tuple2 for a in tuple1
    ]


def filter_positive_element_tuples(list_of_tuples: List[Tuple]) -> List[Tuple]:
    """Filters a list of tuples to keep only those where all elements are positive."""
    return [t for t in list_of_tuples if all(elem >= 0 for elem in t)]


def group_tuples_by_first_element(
    list_of_tuples: List[Tuple],
) -> List[Tuple]:
    """
    Groups tuples from a list that share the same initial element.

    Example: [(5, 6), (5, 7), (6, 8)] -> [(5, 6, 7), (6, 8)]

    Args:
        list_of_tuples: A list of tuples, assumed to be sorted by the first element.

    Returns:
        A new list of grouped tuples.
    """
    # This task is a perfect fit for itertools.groupby, but a loop is also clear.
    if not list_of_tuples:
        return []
    
    result = []
    # Ensure the list is sorted by the key we want to group by.
    sorted_tuples = sorted(list_of_tuples, key=lambda x: x[0])
    
    current_group = list(sorted_tuples[0])
    for i in range(1, len(sorted_tuples)):
        if sorted_tuples[i][0] == current_group[0]:
            current_group.extend(sorted_tuples[i][1:])
        else:
            result.append(tuple(current_group))
            current_group = list(sorted_tuples[i])
    result.append(tuple(current_group))
    
    return result

# ==============================================================================
# General Utility Functions
# ==============================================================================


def find_uncommon_elements(list1: List, list2: List) -> List:
    """Finds elements that are in one list but not both."""
    set1 = set(map(tuple, list1))
    set2 = set(map(tuple, list2))
    # Symmetric difference finds elements in either set, but not both.
    uncommon = set1.symmetric_difference(set2)
    return [list(item) for item in uncommon]


def divide_with_remainder(
    numerator: Union[int, float], denominator: Union[int, float]
) -> Tuple[Union[int, float], Union[int, float]]:
    """Divides two numbers and returns the quotient and remainder."""
    if denominator == 0:
        raise ValueError("Cannot divide by zero.")
    return numerator // denominator, numerator % denominator


def read_file_contents(filepath: str) -> str:
    """Reads the entire content of a file and returns it as a string."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found at '{filepath}'"


def sum_first_n_natural_numbers(n: int) -> int:
    """Calculates the sum of the first N natural numbers using the formula."""
    if n < 0:
        return 0
    return (n * (n + 1)) // 2


def recursive_sum_natural_numbers(n: int) -> int:
    """Calculates the sum of natural numbers up to n using recursion."""
    if n <= 0:
        return 0
    return n + recursive_sum_natural_numbers(n - 1)


def filter_dicts_by_key_value(
    list_of_dicts: List[Dict], key: Any, value: Any
) -> List[Dict]:
    """Filters a list of dictionaries to find those matching a key-value pair."""
    return [d for d in list_of_dicts if d.get(key) == value]


def reverse_sequence_recursively(seq: Sequence) -> Sequence:
    """Reverses a sequence (list, tuple, string) using recursion."""
    if not seq:
        return seq
    return reverse_sequence_recursively(seq[1:]) + seq[0:1]


def square_list_elements(numbers: List[float]) -> List[float]:
    """Returns a new list with the square of each number from the input list."""
    return [x ** 2 for x in numbers]


def guess_the_number_game():
    """A simple, interactive number guessing game."""
    print("Let's play a game! Think of a number between 1 and 5.")
    print("Answer the following questions with 'y' for yes or 'n' for no.")

    if input("Is the number greater than 3? (y/n): ").lower() == 'y':
        # Number is 4 or 5
        if input("Is the number 4? (y/n): ").lower() == 'y':
            print("Great! The number is 4.")
        else:
            print("Great! The number is 5.")
    else:
        # Number is 1, 2, or 3
        if input("Is the number less than 3? (y/n): ").lower() == 'y':
            # Number is 1 or 2
            if input("Is the number 1? (y/n): ").lower() == 'y':
                print("Great! The number is 1.")
            else:
                print("Great! The number is 2.")
        else:
            print("Great! The number is 3.")

# ==============================================================================
# Main execution block
# ==============================================================================


if __name__ == "__main__":
    # Example for: combine_dicts_with_priority
    dict_a = {'Gfg': 1, 'is': 2, 'best': 3}
    dict_b = {'Gfg': 4, 'is': 10, 'for': 7, 'geeks': 12}
    combined = combine_dicts_with_priority(low_prio_dict=dict_a, high_prio_dict=dict_b)
    print(f"Dictionary combined with priority: {combined}")

    # Example for: sum_common_key_values
    dict1 = {'a': 12, 'for': 25, 'c': 9}
    dict2 = {'Geeks': 100, 'geek': 200, 'for': 300}
    summed_dict = sum_common_key_values(dict1, dict2)
    print(f"Dictionary with summed values for common keys: {summed_dict}")

    # Example for: sort_dict_keys_by_value
    my_dict = {'Geeks': 2, 'for': 1, 'CS': 3}
    sorted_keys = sort_dict_keys_by_value(my_dict)
    print(f"Dictionary keys sorted by value: {sorted_keys}")

    # Example for: concatenate_dict_list_values
    list_of_dicts_to_concat = [
        {'tsai': [1, 5, 6, 7], 'good': [9, 6, 2, 10], 'CS': [4, 5, 6]},
        {'tsai': [5, 6, 7, 8], 'CS': [5, 7, 10]},
        {'tsai': [7, 5], 'best': [5, 7]}
    ]
    concatenated = concatenate_dict_list_values(list_of_dicts_to_concat)
    print(f"Concatenated dictionary: {concatenated}")

    # Example for: get_top_n_largest_keys
    int_key_dict = {6: 2, 8: 9, 3: 9, 10: 8}
    top_keys = get_top_n_largest_keys(int_key_dict, n=3)
    print(f"Top 3 largest keys are: {top_keys}")

    # Example for: extract_value_by_key
    list_of_dicts_to_extract = [
        {"Gfg": 3, "b": 7}, {"is": 5, 'a': 10}, {"Best": 9, 'c': 11}
    ]
    extracted_value = extract_value_by_key(list_of_dicts_to_extract, key='Best')
    print(f"The extracted value for key 'Best' is: {extracted_value}")

    # Example for: date_str_to_timestamp
    ts = date_str_to_timestamp("20/01/2020")
    print(f"Timestamp for 20/01/2020: {ts}")

    # Example for: product_of_integers_in_list
    mixed_list = [5, 8, "gfg", 8.5, (5, 7), 'is', 2]
    product = product_of_integers_in_list(mixed_list)
    print(f"Product of integers in mixed list: {product}")
    
    # Example for: remove_even_numbers
    num_list = [11, 5, 17, 18, 23, 50]
    odd_list = remove_even_numbers(num_list)
    print(f"List after removing even numbers: {odd_list}")

    # Example for: count_occurrences
    my_list = [8, 6, 8, 10, 8, 20, 10, 8, 8]
    count_of_8 = count_occurrences(my_list, 8)
    print(f"The number 8 occurred {count_of_8} times.")
    
    # Example for: swap_first_and_last
    list_to_swap = [12, 35, 9, 56, 24]
    swapped = swap_first_and_last(list_to_swap.copy()) # Pass a copy to preserve original
    print(f"List after swapping first and last elements: {swapped}")

    # Example for: count_unique_elements
    input_list = [1, 2, 2, 5, 8, 4, 4, 8]
    unique_count = count_unique_elements(input_list)
    print(f"Number of unique items: {unique_count}")
    
    # Example for: get_sum_and_average
    numbers_for_avg = [4, 5, 1, 2, 9, 7, 10, 8]
    total, avg = get_sum_and_average(numbers_for_avg)
    print(f"Sum = {total}, Average = {avg}")
    
    # Example for: sort_tuples_by_second_item
    tuples_to_sort = [(3, 4), (1, 2), (6, 5), (7, 0)]
    sorted_tups = sort_tuples_by_second_item(tuples_to_sort)
    print(f"Tuples sorted by second item: {sorted_tups}")
    
    # Example for: time_12hr_to_24hr
    time_24 = time_12hr_to_24hr("07:05:45 PM")
    print(f"07:05:45 PM in 24-hour format is: {time_24}")
    
