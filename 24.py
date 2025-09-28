# list_manipulation.py

"""
A collection of Python examples demonstrating common list manipulations,
refactored for clarity, efficiency, and adherence to best practices.
"""

import math
from typing import List, Any, Union, Set

# --- Filtering and Removing Elements ---

def filter_odd_numbers(numbers: List[int]) -> List[int]:
    """
    Removes even numbers from a list by creating a new list containing only odd numbers.

    Note: The original code had a common bug: modifying a list while iterating
    over it (list.remove(ele)). This can lead to skipped elements. Creating a
    new list via a list comprehension is the correct and safe approach.

    Args:
        numbers: A list of integers.

    Returns:
        A new list containing only the odd numbers from the original list.
    """
    return [num for num in numbers if num % 2 != 0]

def remove_elements_by_index_range(data: List[Any], start: int, end: int) -> None:
    """
    Removes elements from a list in-place based on an index range.

    Args:
        data: The list to modify.
        start: The starting index of the slice to remove.
        end: The ending index (exclusive) of the slice to remove.
    """
    del data[start:end]

def remove_unwanted_elements(data: List[Any], unwanted: Set[Any]) -> List[Any]:
    """
    Removes a specific set of unwanted elements from a list.

    Using a set for `unwanted` provides a highly efficient O(1) average time
    complexity for checking membership.

    Args:
        data: The list to filter.
        unwanted: A set of elements to remove.

    Returns:
        A new list without the unwanted elements.
    """
    return [element for element in data if element not in unwanted]

def filter_non_empty_strings(strings: List[str]) -> List[str]:
    """
    Removes all empty or whitespace-only strings from a list of strings.

    The `str.strip()` method removes leading/trailing whitespace. An empty
    string evaluates to False in a boolean context.

    Args:
        strings: A list of strings.

    Returns:
        A new list containing only non-empty strings.
    """
    return [s for s in strings if s.strip()]

# --- List Transformations and Calculations ---

def get_cumulative_sum(numbers: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Calculates the cumulative sum of a list of numbers.

    Args:
        numbers: A list of numbers (integers or floats).

    Returns:
        A list where each element is the cumulative sum up to that point.
    """
    # Using a simple loop is more readable and often more efficient
    # than the original list comprehension with repeated sum() calls.
    cumulative_list = []
    current_sum = 0
    for num in numbers:
        current_sum += num
        cumulative_list.append(current_sum)
    return cumulative_list

def swap_first_and_last(data: List[Any]) -> List[Any]:
    """
    Swaps the first and last elements of a list.

    This implementation uses tuple unpacking for a concise and Pythonic swap.
    It also handles lists with fewer than two elements gracefully.

    Args:
        data: The list to modify.

    Returns:
        The list with the first and last elements swapped.
    """
    if len(data) >= 2:
        data[0], data[-1] = data[-1], data[0]
    return data

def reverse_list(data: List[Any]) -> List[Any]:
    """
    Returns a reversed copy of a list.

    Args:
        data: The list to reverse.

    Returns:
        A new list containing the elements of the original list in reverse order.
    """
    # Slicing `[::-1]` is a common and concise idiom for reversing a list.
    return data[::-1]

# --- Main execution block ---
if __name__ == "__main__":
    # Example 1: Remove even numbers
    print("--- 1. Remove even numbers ---")
    my_numbers = [11, 5, 17, 18, 23, 50]
    odd_numbers = filter_odd_numbers(my_numbers)
    print(f"Original list: {my_numbers}")
    print(f"List after removing even numbers: {odd_numbers}\n")

    # Example 2: Remove elements by index range
    print("--- 2. Remove elements from index 1 to 4 ---")
    my_list = [11, 5, 17, 18, 23, 50]
    print(f"Original list: {my_list}")
    remove_elements_by_index_range(my_list, 1, 5) # Note: index 5 is not included
    print(f"List after removing elements[1:5]: {my_list}\n")

    # Example 3: Remove a specific set of numbers
    print("--- 3. Remove unwanted numbers (11 and 18) ---")
    my_numbers = [11, 5, 17, 18, 23, 50]
    unwanted = {11, 18}
    filtered_list = remove_unwanted_elements(my_numbers, unwanted)
    print(f"Original list: {my_numbers}")
    print(f"List after removing {unwanted}: {filtered_list}\n")

    # Example 4: Remove empty strings
    print("--- 4. Remove empty strings from a list ---")
    string_list = ['gfg', '   ', ' ', 'is', '            ', 'best']
    print(f"Original list: {string_list}")
    filtered_strings = filter_non_empty_strings(string_list)
    print(f"List after filtering non-empty strings: {filtered_strings}\n")

    # Example 5: Get cumulative sum
    print("--- 5. Get cumulative sum of a list ---")
    numbers_for_sum = [10, 20, 30, 40, 50]
    cumulative_sums = get_cumulative_sum(numbers_for_sum)
    print(f"Original list: {numbers_for_sum}")
    print(f"Cumulative sum list: {cumulative_sums}\n")
    
    # Example 6: Swap first and last elements
    print("--- 6. Swap first and last elements ---")
    my_list_to_swap = [12, 35, 9, 56, 24]
    print(f"Original list: {my_list_to_swap}")
    swapped_list = swap_first_and_last(my_list_to_swap)
    print(f"Swapped list: {swapped_list}\n")
    
    # Example 7: Reverse a list
    print("--- 7. Reverse a list ---")
    list_to_reverse = [10, 11, 12, 13, 14, 15]
    print(f"Original list: {list_to_reverse}")
    reversed_list = reverse_list(list_to_reverse)
    print(f"Reversed list: {reversed_list}\n")


# list_analysis.py

"""
A collection of Python examples for analyzing list data, such as
finding elements, counting occurrences, and calculating statistics.
"""

from typing import List, Any, Union

# --- Searching and Finding ---

def check_presence(element: Any, data: List[Any]) -> bool:
    """
    Checks if an element is present in a list.

    Args:
        element: The element to search for.
        data: The list to search within.

    Returns:
        True if the element is in the list, False otherwise.
    """
    return element in data

def find_distance_between_even_numbers(numbers: List[int]) -> int:
    """
    Calculates the distance between the first and last occurrence of an even number.

    Args:
        numbers: A list of integers.

    Returns:
        The index distance between the first and last even number.
        Returns 0 if fewer than two even numbers are found.
    """
    even_indices = [i for i, num in enumerate(numbers) if num % 2 == 0]

    if len(even_indices) < 2:
        return 0
    
    return even_indices[-1] - even_indices[0]

# --- Counting and Statistics ---

def count_element_occurrences(data: List[Any], element: Any) -> int:
    """
    Counts the number of times an element appears in a list.

    Uses the built-in `list.count()` method, which is efficient and readable.

    Args:
        data: The list to search within.
        element: The element to count.

    Returns:
        The number of occurrences of the element.
    """
    return data.count(element)

def count_unique_elements(data: List[Any]) -> int:
    """
    Counts the number of unique elements in a list.

    Leverages the `set` data structure, which automatically handles uniqueness.

    Args:
        data: A list of items.

    Returns:
        The count of unique items.
    """
    return len(set(data))

def calculate_sum_and_average(numbers: List[Union[int, float]]) -> tuple[Union[int, float], float]:
    """
    Calculates the sum and average of a list of numbers.

    Uses the built-in `sum()` and `len()` functions for simplicity and efficiency.

    Args:
        numbers: A list of numbers.

    Returns:
        A tuple containing the sum and the average. Returns (0, 0.0) for an empty list.
    """
    if not numbers:
        return 0, 0.0
        
    total_sum = sum(numbers)
    average = total_sum / len(numbers)
    return total_sum, average

# --- Main execution block ---
if __name__ == "__main__":
    # Example 1: Check if "hello" is in the list
    print('--- 1. Check for element presence ---')
    mixed_list = [1, 2.0, 'hello', 'have', 'a', 'good', 'day']
    search_term = 'hello'
    if check_presence(search_term, mixed_list):
        print(f"'{search_term}' is present in the list.\n")
    else:
        print(f"'{search_term}' is not present in the list.\n")

    # Example 2: Distance between first and last even number
    print('--- 2. Distance between first and last even number ---')
    num_list = [1, 3, 7, 4, 7, 2, 9, 1, 10, 11]
    distance = find_distance_between_even_numbers(num_list)
    print(f"The list is: {num_list}")
    print(f"Distance between first and last even elements: {distance}\n")
    
    # Example 3: Count occurrences of 8
    print('--- 3. Count element occurrences ---')
    numbers_to_count = [8, 6, 8, 10, 8, 20, 10, 8, 8]
    element_to_count = 8
    occurrences = count_element_occurrences(numbers_to_count, element_to_count)
    print(f"The list is: {numbers_to_count}")
    print(f"'{element_to_count}' has occurred {occurrences} times.\n")

    # Example 4: Count unique elements
    print('--- 4. Count unique elements ---')
    list_with_duplicates = [1, 2, 2, 5, 8, 4, 4, 8]
    unique_count = count_unique_elements(list_with_duplicates)
    print(f"The list is: {list_with_duplicates}")
    print(f"Number of unique items: {unique_count}\n")
    
    # Example 5: Calculate sum and average
    print('--- 5. Calculate sum and average ---')
    numbers_for_stats = [4, 5, 1, 2, 9, 7, 10, 8]
    total, avg = calculate_sum_and_average(numbers_for_stats)
    print(f"The list is: {numbers_for_stats}")
    print(f"Sum = {total}")
    print(f"Average = {avg}\n")


# tuple_operations.py

"""
A collection of Python examples demonstrating operations on tuples and lists of tuples.
"""

from typing import List, Tuple, Any

def filter_tuples_by_length(tuples_list: List[Tuple], length_to_remove: int) -> List[Tuple]:
    """
    Removes tuples of a specific length from a list of tuples.

    Args:
        tuples_list: A list of tuples.
        length_to_remove: The length of tuples that should be removed.

    Returns:
        A new list containing tuples that do not match the specified length.
    """
    return [tup for tup in tuples_list if len(tup) != length_to_remove]

def create_number_and_cube_tuples(numbers: List[int]) -> List[Tuple[int, int]]:
    """
    Creates a list of tuples, where each tuple contains a number and its cube.

    Args:
        numbers: A list of integers.

    Returns:
        A list of (number, number^3) tuples.
    """
    return [(num, num**3) for num in numbers]

def swap_tuple_elements(tuples_list: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
    """
    Swaps the elements in each tuple within a list of 2-element tuples.

    Args:
        tuples_list: A list of tuples, where each tuple has two elements.

    Returns:
        A new list with the elements of each tuple swapped.
    """
    return [(b, a) for a, b in tuples_list]

def sort_tuples_by_second_item(tuples_list: List[Tuple[Any, Any]]) -> List[Tuple[Any, Any]]:
    """
    Sorts a list of tuples based on the second element of each tuple.

    The original implementation used an inefficient manual bubble sort. Using the
    built-in `sorted()` function with a lambda key is the standard, efficient,
    and Pythonic way to perform custom sorting.

    Args:
        tuples_list: A list of tuples to be sorted.

    Returns:
        A new list of tuples, sorted by the second item.
    """
    return sorted(tuples_list, key=lambda x: x[1])

def get_all_pair_combinations(tuple1: Tuple, tuple2: Tuple) -> List[Tuple]:
    """
    Generates all pair combinations from two tuples.

    Args:
        tuple1: The first tuple.
        tuple2: The second tuple.

    Returns:
        A list of tuples representing all combinations.
    """
    # The original code concatenated two list comprehensions.
    # We can use itertools for more complex combinations, but for this
    # simple case, the original logic is clear enough.
    combinations1 = [(a, b) for a in tuple1 for b in tuple2]
    combinations2 = [(a, b) for a in tuple2 for b in tuple1]
    return combinations1 + combinations2

def filter_positive_tuples(tuples_list: List[Tuple[Union[int, float], ...]]) -> List[Tuple]:
    """
    Filters a list of tuples, keeping only those where all elements are non-negative.

    Args:
        tuples_list: A list of tuples containing numbers.

    Returns:
        A new list containing only the "positive" tuples.
    """
    return [tup for tup in tuples_list if all(elem >= 0 for elem in tup)]

def group_tuples_by_first_element(tuples_list: List[Tuple]) -> List[Tuple]:
    """
    Joins tuples from a list if they have the same initial element.

    Example: [(5, 6), (5, 7), (6, 8)] -> [(5, 6, 7), (6, 8)]

    Args:
        tuples_list: A sorted or structured list of tuples.

    Returns:
        A list of grouped tuples.
    """
    if not tuples_list:
        return []

    # Using a dictionary is a more robust and readable way to group items.
    from collections import defaultdict
    grouped = defaultdict(list)
    for key, *values in tuples_list:
        grouped[key].extend(values)

    # Reconstruct the list of tuples
    result = []
    for key, values in grouped.items():
        result.append(tuple([key] + values))
        
    return result

# --- Main execution block ---
if __name__ == "__main__":
    # Example 1: Remove tuples of length 1
    print("--- 1. Remove tuples of length 1 ---")
    list_of_tuples = [(4, 5), (4,), (8, 6, 7), (1,), (3, 4, 6, 7)]
    filtered_tuples = filter_tuples_by_length(list_of_tuples, 1)
    print(f"Original list: {list_of_tuples}")
    print(f"Filtered list: {filtered_tuples}\n")

    # Example 2: Create number-cube tuples
    print("--- 2. Create number and cube tuples ---")
    numbers = [1, 2, 5, 6]
    cube_tuples = create_number_and_cube_tuples(numbers)
    print(f"Original numbers: {numbers}")
    print(f"Resultant tuples: {cube_tuples}\n")

    # Example 3: Swap tuple elements
    print("--- 3. Swap tuple elements ---")
    tuples_to_swap = [(3, 4), (6, 5), (7, 8)]
    swapped = swap_tuple_elements(tuples_to_swap)
    print(f"Original list: {tuples_to_swap}")
    print(f"Swapped tuple list: {swapped}\n")
    
    # Example 4: Sort tuples by second item
    print("--- 4. Sort tuples by second item ---")
    unsorted_tuples = [('a', 3), ('b', 1), ('c', 2)]
    sorted_list = sort_tuples_by_second_item(unsorted_tuples)
    print(f"Unsorted list: {unsorted_tuples}")
    print(f"Sorted list: {sorted_list}\n")

    # Example 5: Get all pair combinations
    print("--- 5. Get all pair combinations ---")
    tuple_a = (4, 5)
    tuple_b = (7, 8)
    combinations = get_all_pair_combinations(tuple_a, tuple_b)
    print(f"Tuples: {tuple_a} and {tuple_b}")
    print(f"All combinations: {combinations}\n")
    
    # Example 6: Filter positive tuples
    print("--- 6. Filter for positive tuples ---")
    mixed_tuples = [(4, 5, 9), (-3, 2, 3), (-3, 5, 6), (4, 6)]
    positive_tuples = filter_positive_tuples(mixed_tuples)
    print(f"Original list: {mixed_tuples}")
    print(f"Positive element tuples: {positive_tuples}\n")

    # Example 7: Group tuples by first element
    print("--- 7. Group tuples by first element ---")
    tuples_to_group = [(5, 6), (5, 7), (6, 8), (6, 10), (7, 13)]
    grouped = group_tuples_by_first_element(tuples_to_group)
    print(f"Original list: {tuples_to_group}")
    print(f"Grouped list: {grouped}\n")



# functions_and_classes.py

"""
A collection of Python examples covering functions, classes,
exception handling, and other language features.
"""

import os
from typing import Any, List, Union

# --- Classes ---

class TreeNode:
    """Represents a node in a tree structure."""
    def __init__(self, data: Any):
        """
        Initializes a TreeNode.

        Args:
            data: The data to be stored in the node.
        """
        self.data = data
        self.parent: Union[TreeNode, None] = None
        self.children: List[TreeNode] = []

    def add_child(self, child: 'TreeNode') -> None:
        """
        Adds a child node to the current node.

        Args:
            child: The TreeNode instance to be added as a child.
        """
        child.parent = self
        self.children.append(child)

    def __repr__(self) -> str:
        """Provides a developer-friendly representation of the node."""
        return f"TreeNode(data={self.data!r})"

class Person:
    """A simple class representing a person."""
    def __init__(self, first_name: str, last_name: str):
        self.first_name = first_name
        self.last_name = last_name

    def print_name(self) -> None:
        """Prints the person's full name."""
        print(f"{self.first_name} {self.last_name}")

class Student(Person):
    """A class representing a student, inheriting from Person."""
    # 'pass' indicates that this class inherits all methods from Person
    # without adding any new ones.
    pass

# --- Exception Handling ---

def read_file_safely(filepath: str) -> str:
    """
    Reads the content of a text file.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        IOError: For other reading errors.

    Args:
        filepath: The path to the file.

    Returns:
        The content of the file as a string.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        raise
    except IOError as e:
        print(f"Error reading file at '{filepath}': {e}")
        raise

def check_value_greater_than_ten(value: int) -> int:
    """
    Checks if a value is 10 or greater.

    Raises:
        ValueError: If the value is less than 10.
    
    Args:
        value: The integer to check.

    Returns:
        The original value if it's valid.
    """
    if value < 10:
        raise ValueError('Value must be 10 or greater.')
    return value

# --- Utility Functions ---

def join_path_components(parent_dir: str, child_dir: str) -> str:
    """
    Safely joins two path components into a single path string.

    Args:
        parent_dir: The parent directory path.
        child_dir: The child directory or filename.

    Returns:
        A new string representing the combined path.
    """
    return os.path.join(parent_dir, child_dir)

def get_type_name(obj: Any) -> str:
    """
    Returns the type of the given object as a string.

    Args:
        obj: The object to inspect.

    Returns:
        A string like "list", "dict", "tuple", etc.
    """
    # Using a dictionary mapping is a clean way to handle multiple type checks.
    type_map = {
        list: "list",
        dict: "dict",
        tuple: "tuple",
        set: "set",
        str: "string"
    }
    # .get() provides a default value if the type is not in our map.
    return type_map.get(type(obj), "unknown")
    
def time_str_to_24_hour(time_str: str) -> str:
    """
    Converts a time string from 12-hour format (e.g., "07:05:45 PM")
    to 24-hour format (e.g., "19:05:45").

    This is better handled by Python's `datetime` module for robustness.
    
    Args:
        time_str: Time in 12-hour format with AM/PM.

    Returns:
        Time in 24-hour format.
    """
    from datetime import datetime
    
    # Let the datetime module parse the 12-hour format string
    time_obj = datetime.strptime(time_str, '%I:%M:%S %p')
    
    # Format it back into a 24-hour format string
    return time_obj.strftime('%H:%M:%S')

# --- Main execution block ---
if __name__ == "__main__":
    # Example 1: Class and inheritance
    print("--- 1. Class and Inheritance ---")
    student = Student("John", "Doe")
    student.print_name()
    print("")

    # Example 2: Safe file reading
    print("--- 2. Safe File Reading ---")
    try:
        # Create a dummy file to read
        with open("test.txt", "w") as f:
            f.write("Hello, World!")
        content = read_file_safely("test.txt")
        print(f"File content: '{content}'")
        # Demonstrate exception
        read_file_safely("non_existent_file.txt")
    except FileNotFoundError:
        print("Caught the expected FileNotFoundError.\n")
    finally:
        # Clean up the dummy file
        if os.path.exists("test.txt"):
            os.remove("test.txt")

    # Example 3: Value checking with exceptions
    print("--- 3. Value Checking with Exceptions ---")
    try:
        check_value_greater_than_ten(15)
        print("check_value_greater_than_ten(15) passed.")
        check_value_greater_than_ten(9)
    except ValueError as e:
        print(f"Caught expected error: {e}\n")

    # Example 4: Type checking
    print("--- 4. Type Checking ---")
    print(f"Type of [1, 2]: {get_type_name([1, 2])}")
    print(f"Type of {{'a': 1}}: {get_type_name({'a': 1})}")
    print(f"Type of (1,): {get_type_name((1,))}")
    print(f"Type of 123: {get_type_name(123)}\n")
    
    # Example 5: 12-hour to 24-hour time conversion
    print("--- 5. Time Format Conversion ---")
    time12h = "07:05:45 PM"
    time24h = time_str_to_24_hour(time12h)
    print(f"'{time12h}' in 24-hour format is '{time24h}'")
    time12h_am = "12:00:00 AM"
    time24h_am = time_str_to_24_hour(time12h_am)
    print(f"'{time12h_am}' in 24-hour format is '{time24h_am}'\n")
