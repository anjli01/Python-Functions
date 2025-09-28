"""
A collection of refactored Python code examples following best practices.

This script demonstrates solutions to common programming problems and showcases
various Python features. Each example is encapsulated in a well-documented
function with type hints.

Key improvements over the original script:
- Encapsulation in functions for reusability and clarity.
- Descriptive variable and function names (PEP 8 compliance).
- Docstrings explaining the purpose, arguments, and return values of each function.
- Type hints for improved readability and static analysis.
- Separation of logic (in functions) from presentation (in the main block).
- More efficient and "Pythonic" implementations where applicable.
- A main execution block (`if __name__ == "__main__":`) to run examples.
"""

import json
import random
import re
import secrets
import string
import time
import uuid
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

# For numpy examples, which are common but not in the standard library.
# To run this part, you would need to `pip install numpy`.
try:
    import numpy as np
except ImportError:
    np = None

# ==============================================================================
# 1. ALGORITHM AND DATA STRUCTURE EXAMPLES
# ==============================================================================

def find_longest_substring_length(s: str) -> int:
    """
    Finds the length of the longest substring without repeating characters.

    This implementation uses a sliding window approach for O(n) time complexity,
    which is more efficient than the original O(n^2) brute-force method.

    Args:
        s: The input string.

    Returns:
        The length of the longest substring without repeating characters.
    """
    if not s:
        return 0
    
    char_index_map = {}
    max_length = 0
    start = 0
    
    for i, char in enumerate(s):
        if char in char_index_map and char_index_map[char] >= start:
            start = char_index_map[char] + 1
        char_index_map[char] = i
        max_length = max(max_length, i - start + 1)
        
    return max_length

def find_two_sum_indices(nums: List[int], target: int) -> Optional[Tuple[int, int]]:
    """
    Returns indices of two numbers in a list that add up to a specific target.

    This implementation uses a hash map (dictionary) for O(n) time complexity,
    which is a significant improvement over the original O(n^2) nested loop.

    Args:
        nums: A list of integers.
        target: The target sum.

    Returns:
        A tuple containing the indices of the two numbers, or None if not found.
    """
    num_to_index = {}
    for index, num in enumerate(nums):
        complement = target - num
        if complement in num_to_index:
            return num_to_index[complement], index
        num_to_index[num] = index
    return None

def summarize_ranges(nums: List[int]) -> List[str]:
    """
    Given a sorted integer list without duplicates, returns the summary of its ranges.
    
    Example: [0,1,2,4,5,7] -> ["0->2", "4->5", "7"]

    Args:
        nums: A sorted list of unique integers.

    Returns:
        A list of strings representing the summarized ranges.
    """
    if not nums:
        return []

    ranges = []
    start = 0
    while start < len(nums):
        end = start
        while end + 1 < len(nums) and nums[end + 1] == nums[end] + 1:
            end += 1
        
        if end == start:
            ranges.append(f"{nums[start]}")
        else:
            ranges.append(f"{nums[start]}->{nums[end]}")
        start = end + 1
        
    return ranges

def rotate_array_right(nums: List[Any], k: int) -> List[Any]:
    """
    Rotates a list to the right by k steps.

    This implementation uses slicing, which is more concise and Pythonic.
    It also handles cases where k is larger than the list length.

    Args:
        nums: The list to rotate.
        k: The number of steps to rotate.

    Returns:
        The rotated list.
    """
    if not nums:
        return []
    
    k = k % len(nums)
    return nums[-k:] + nums[:-k]

def find_missing_element(arr1: List[int], arr2: List[int]) -> Optional[int]:
    """
    Finds the missing element from a shuffled list.

    Assumes arr2 is a shuffled version of arr1 with one element removed.
    The sum-based approach is simple and efficient for non-negative integers.

    Args:
        arr1: The original list of non-negative integers.
        arr2: The shuffled list with one element missing.

    Returns:
        The missing integer, or None if lists are identical.
    """
    if sum(arr1) == sum(arr2):
        return None
    return sum(arr1) - sum(arr2)

def merge_overlapping_intervals(intervals: List[List[int]]) -> List[List[int]]:
    """
    Merges overlapping intervals in a list of [start, end] pairs.
    
    The input list is assumed to be sorted by the start time of the interval.
    This implementation builds a new list, which is cleaner than modifying
    the list in-place while iterating.

    Args:
        intervals: A list of intervals, sorted by start time.

    Returns:
        A new list of merged intervals.
    """
    if not intervals:
        return []

    merged = [intervals[0]]
    for current_start, current_end in intervals[1:]:
        last_start, last_end = merged[-1]
        
        if current_start <= last_end:
            # Overlap detected, merge by updating the end of the last interval
            merged[-1][1] = max(last_end, current_end)
        else:
            # No overlap, add the new interval
            merged.append([current_start, current_end])
            
    return merged
    
def bubble_sort(nums: List[int]) -> List[int]:
    """
    Sorts a list of numbers using the bubble sort algorithm.

    Args:
        nums: A list of numbers.

    Returns:
        The sorted list.
    """
    n = len(nums)
    # Create a copy to avoid modifying the original list
    sorted_nums = nums[:] 
    for i in range(n):
        # Last i elements are already in place
        for j in range(0, n - i - 1):
            if sorted_nums[j] > sorted_nums[j+1]:
                sorted_nums[j], sorted_nums[j+1] = sorted_nums[j+1], sorted_nums[j]
    return sorted_nums

# ==============================================================================
# 2. LIST, TUPLE, SET, AND DICTIONARY OPERATIONS
# ==============================================================================

def chunk_and_reverse_list(data: List[Any], num_chunks: int = 3) -> List[List[Any]]:
    """
    Slices a list into `num_chunks` equal chunks and reverses each chunk.
    Handles lists where the length is not perfectly divisible by `num_chunks`.

    Args:
        data: The input list.
        num_chunks: The number of chunks to create.

    Returns:
        A list of reversed chunks.
    """
    if not data or num_chunks <= 0:
        return []
    
    chunk_size = len(data) // num_chunks
    chunks = []
    for i in range(0, len(data), chunk_size):
        chunk = data[i:i + chunk_size]
        chunks.append(chunk[::-1]) # [::-1] is a concise way to reverse
    
    # This example specifically asked for 3 chunks, so we'll ensure the last
    # part is merged if the division wasn't even.
    # A more generic chunking function would just return the chunks as is.
    if len(chunks) > num_chunks:
        chunks[num_chunks-1].extend(item for sublist in chunks[num_chunks:] for item in sublist)
        return chunks[:num_chunks]
        
    return chunks

def find_middle_element(data: List[Any]) -> Any:
    """
    Finds the middle element of a list.
    For even-length lists, it returns the first of the two middle elements.

    Args:
        data: The input list.

    Returns:
        The middle element.
    """
    if not data:
        return None
    return data[len(data) // 2]
    
def count_occurrences_in_list(data: List[Any]) -> Dict[Any, int]:
    """
    Counts the occurrences of each element in a list and returns a dictionary.
    
    Uses `collections.Counter` for a more efficient and Pythonic solution.
    
    Args:
        data: The input list.
        
    Returns:
        A dictionary mapping each element to its count.
    """
    return Counter(data)

def remove_all_occurrences(data: List[Any], value_to_remove: Any) -> List[Any]:
    """
    Removes all occurrences of a specific value from a list.
    
    Uses a list comprehension for a clean, functional approach.
    
    Args:
        data: The input list.
        value_to_remove: The value to remove from the list.
        
    Returns:
        A new list with all occurrences of the value removed.
    """
    return [item for item in data if item != value_to_remove]
    
def get_unique_values_from_dict(data: Dict[Any, Any]) -> List[Any]:
    """
    Gets all unique values from a dictionary and returns them in a list.

    Args:
        data: The input dictionary.

    Returns:
        A list of unique values.
    """
    return list(set(data.values()))


# ==============================================================================
# 3. NUMBER AND MATH OPERATIONS
# ==============================================================================

def calculate_exponent(base: float, exponent: float) -> float:
    """Calculates the exponent of a number."""
    return base ** exponent

def get_remainder(dividend: int, divisor: int) -> int:
    """Calculates the remainder of a division."""
    if divisor == 0:
        raise ValueError("Divisor cannot be zero.")
    return dividend % divisor

def reverse_integer(num: int) -> int:
    """Reverses the digits of a non-negative integer."""
    if num < 0:
        raise ValueError("Input must be a non-negative integer.")
        
    reversed_num = 0
    while num > 0:
        remainder = num % 10
        reversed_num = (reversed_num * 10) + remainder
        num = num // 10
    return reversed_num

def sum_digits_recursive(num: int) -> int:
    """Computes the sum of digits in a non-negative integer using recursion."""
    if num < 0:
        raise ValueError("Input must be a non-negative integer.")
    if num == 0:
        return 0
    return (num % 10) + sum_digits_recursive(num // 10)

def generate_fibonacci_up_to(limit: int) -> List[int]:
    """Generates the Fibonacci series up to a given limit."""
    result = []
    a, b = 0, 1
    while a < limit:
        result.append(a)
        a, b = b, a + b
    return result

def multiply_matrices(matrix_a: List[List[float]], matrix_b: List[List[float]]) -> List[List[float]]:
    """
    Multiplies two matrices.
    Note: For serious matrix operations, using a library like NumPy is highly recommended.
    """
    rows_a = len(matrix_a)
    cols_a = len(matrix_a[0])
    rows_b = len(matrix_b)
    cols_b = len(matrix_b[0])

    if cols_a != rows_b:
        raise ValueError("Number of columns in Matrix A must equal number of rows in Matrix B.")

    # Initialize result matrix with zeros
    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]

    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a): # or range(rows_b)
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result

# ==============================================================================
# 4. STRING OPERATIONS
# ==============================================================================

def append_string_in_middle(s1: str, s2: str) -> str:
    """Appends string s2 in the middle of string s1."""
    middle_index = len(s1) // 2
    return s1[:middle_index] + s2 + s1[middle_index:]

def sort_chars_by_case(s: str) -> str:
    """Arranges string characters such that lowercase letters come first."""
    lower_chars = [char for char in s if char.islower()]
    upper_chars = [char for char in s if char.isupper()]
    return "".join(lower_chars + upper_chars)

def get_sum_and_avg_of_digits_in_string(s: str) -> Tuple[int, float]:
    """
    Returns the sum and average of digits that appear in a string.
    """
    digits = [int(num) for num in re.findall(r'\d', s)]
    if not digits:
        return 0, 0.0
    
    total = sum(digits)
    average = total / len(digits)
    return total, average

def remove_punctuation(s: str) -> str:
    """Removes all punctuation from a string."""
    return s.translate(str.maketrans('', '', string.punctuation))

def extract_integers_from_string(s: str) -> str:
    """Removes all characters other than integers from a string."""
    return "".join(filter(str.isdigit, s))

# ==============================================================================
# 5. JSON AND FILE OPERATIONS
# ==============================================================================

def pretty_print_json_string(json_string: str) -> str:
    """
    Parses a JSON string and returns a pretty-printed version.
    """
    try:
        parsed_json = json.loads(json_string)
        return json.dumps(parsed_json, indent=4, sort_keys=True)
    except json.JSONDecodeError:
        return "Invalid JSON string provided."

def save_dict_to_json_file(data: Dict[Any, Any], file_path: Union[str, Path]):
    """
    Writes a dictionary to a file in JSON format with pretty printing.
    """
    path = Path(file_path)
    with path.open('w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, sort_keys=True)
    print(f"Successfully wrote JSON data to {path}")

# ==============================================================================
# 6. DATETIME OPERATIONS
# ==============================================================================

def string_to_datetime(date_string: str, date_format: str) -> Optional[datetime]:
    """Converts a string into a datetime object given a format."""
    try:
        return datetime.strptime(date_string, date_format)
    except ValueError:
        print(f"Error: date_string '{date_string}' does not match format '{date_format}'")
        return None

def get_date_one_week_ago(from_date: datetime) -> datetime:
    """Subtracts one week (7 days) from a given date."""
    return from_date - timedelta(days=7)

def get_day_of_week(date_obj: datetime) -> str:
    """Finds the day of the week (e.g., 'Monday') of a given date."""
    return date_obj.strftime('%A')
    
def days_between_dates(date1: datetime, date2: datetime) -> int:
    """Calculates the absolute number of days between two dates."""
    return abs((date2 - date1).days)

# ==============================================================================
# 7. RANDOM DATA GENERATION
# ==============================================================================

def generate_random_string(length: int) -> str:
    """Generates a random string of a specified length."""
    letters = string.ascii_letters
    return ''.join(random.choice(letters) for _ in range(length))

def get_random_date(start_date_str: str, end_date_str: str, date_format: str = '%m/%d/%Y') -> str:
    """Generates a random date between two given dates."""
    start_time = time.mktime(time.strptime(start_date_str, date_format))
    end_time = time.mktime(time.strptime(end_date_str, date_format))
    
    random_time = start_time + random.random() * (end_time - start_time)
    return time.strftime(date_format, time.localtime(random_time))
    
def generate_secure_random_int(upper_bound: int) -> int:
    """Generates a cryptographically secure random integer below an upper bound."""
    return secrets.randbelow(upper_bound)


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    """
    Main function to demonstrate the refactored code examples.
    """
    print("="*50)
    print("DEMONSTRATING REFACTORED PYTHON EXAMPLES")
    print("="*50)

    # --- 1. Algorithm Examples ---
    print("\n--- 1. Longest Substring Without Repeating Characters ---")
    test_str = "piyushjain"
    length = find_longest_substring_length(test_str)
    print(f"Input string: '{test_str}'")
    print(f"Length of longest non-repeating substring: {length}") # Expected: 7 ('yushjain')

    print("\n--- 2. Two Sum ---")
    nums = [2, 7, 11, 15]
    target = 26
    indices = find_two_sum_indices(nums, target)
    print(f"Input list: {nums}, Target: {target}")
    print(f"Indices: {indices}") # Expected: (2, 3)

    print("\n--- 3. Summary Ranges ---")
    sorted_nums = [0, 1, 2, 4, 5, 7, 9, 10]
    ranges = summarize_ranges(sorted_nums)
    print(f"Input list: {sorted_nums}")
    print(f"Summarized ranges: {ranges}") # Expected: ['0->2', '4->5', '7', '9->10']

    print("\n--- 4. Rotate Array ---")
    array_to_rotate = [1, 2, 3, 4, 5, 6, 7]
    steps = 3
    rotated = rotate_array_right(array_to_rotate, steps)
    print(f"Original: {array_to_rotate}, Steps: {steps}")
    print(f"Rotated: {rotated}") # Expected: [5, 6, 7, 1, 2, 3, 4]

    print("\n--- 5. Find Missing Element ---")
    arr1 = [1, 2, 3, 4, 5, 6, 7]
    arr2 = [3, 7, 2, 1, 4, 6]
    missing = find_missing_element(arr1, arr2)
    print(f"Array 1: {arr1}\nArray 2: {arr2}")
    print(f"Missing element: {missing}") # Expected: 5

    print("\n--- 6. Merge Overlapping Intervals ---")
    intervals = [[1, 3], [2, 6], [8, 10], [15, 18]]
    merged = merge_overlapping_intervals(intervals)
    print(f"Original intervals: {intervals}")
    print(f"Merged intervals: {merged}") # Expected: [[1, 6], [8, 10], [15, 18]]
    
    print("\n--- 7. Bubble Sort ---")
    unsorted_list = [5, 1, 4, 2, 8]
    sorted_list = bubble_sort(unsorted_list)
    print(f"Unsorted list: {unsorted_list}")
    print(f"Sorted list: {sorted_list}")

    # --- 2. List, Set, Dictionary Operations ---
    print("\n--- 8. Chunk and Reverse List ---")
    sample_list = [11, 45, 8, 23, 14, 12, 78, 45, 89]
    rev_chunks = chunk_and_reverse_list(sample_list, 3)
    print(f"Original list: {sample_list}")
    print(f"Reversed chunks: {rev_chunks}") # Expected: [[8, 45, 11], [12, 14, 23], [89, 45, 78]]

    print("\n--- 9. Find Middle Element ---")
    print(f"Middle of [1, 2, 3, 4, 5] is: {find_middle_element([1, 2, 3, 4, 5])}")
    print(f"Middle of [1, 2, 3, 4, 5, 6] is: {find_middle_element([1, 2, 3, 4, 5, 6])}")

    print("\n--- 10. Count Occurrences in a List ---")
    list_to_count = [11, 45, 8, 11, 23, 45, 23, 45, 89]
    counts = count_occurrences_in_list(list_to_count)
    print(f"List: {list_to_count}")
    print(f"Counts: {counts}")

    # --- 3. Number and Math Operations ---
    print("\n--- 11. Reverse Integer ---")
    num_to_reverse = 1367891
    rev_num = reverse_integer(num_to_reverse)
    print(f"Original number: {num_to_reverse}")
    print(f"Reversed number: {rev_num}") # Expected: 1987631

    print("\n--- 12. Sum of Digits (Recursive) ---")
    num_to_sum = 12345
    digit_sum = sum_digits_recursive(num_to_sum)
    print(f"Number: {num_to_sum}")
    print(f"Sum of digits: {digit_sum}") # Expected: 15

    print("\n--- 13. Matrix Multiplication ---")
    matrix_x = [[12, 7, 3], [4, 5, 6], [7, 8, 9]]
    matrix_y = [[5, 8, 1, 2], [6, 7, 3, 0], [4, 5, 9, 1]]
    try:
        matrix_product = multiply_matrices(matrix_x, matrix_y)
        print(f"Matrix X:\n{matrix_x}")
        print(f"Matrix Y:\n{matrix_y}")
        print(f"Product:\n{matrix_product}")
    except ValueError as e:
        print(e)
        
    if np:
        print("\n--- (Numpy) Matrix Multiplication ---")
        np_product = np.dot(np.array(matrix_x), np.array(matrix_y))
        print("For comparison, the product using NumPy:")
        print(np_product)

    # --- 4. String Operations ---
    print("\n--- 14. Append String in Middle ---")
    result_str = append_string_in_middle("Ault", "Kelly")
    print(f"Result: {result_str}") # Expected: AuKellylt

    print("\n--- 15. Sort Chars by Case ---")
    mixed_case_str = "PyNaTive"
    sorted_case_str = sort_chars_by_case(mixed_case_str)
    print(f"Original: {mixed_case_str}")
    print(f"Sorted by case: {sorted_case_str}") # Expected: yativePNT
    
    print("\n--- 16. Remove Punctuation ---")
    punc_str = "/*Jon is @developer & musician"
    clean_str = remove_punctuation(punc_str)
    print(f"Original: '{punc_str}'")
    print(f"Cleaned: '{clean_str}'")

    # --- 5. JSON Operations ---
    print("\n--- 17. Pretty Print JSON ---")
    person_string = '{"name": "Bob", "languages": "English", "numbers": [2, 1.6, null]}'
    pretty_json = pretty_print_json_string(person_string)
    print(f"Original JSON string: {person_string}")
    print("Pretty-printed JSON:")
    print(pretty_json)
    
    print("\n--- 18. Save Dictionary to JSON file ---")
    person_dict = {
        "name": "Alice",
        "age": 32,
        "married": True,
        "languages": ["English", "French"]
    }
    save_dict_to_json_file(person_dict, "person.json")

    # --- 6. Datetime Operations ---
    print("\n--- 19. String to Datetime ---")
    date_str = "Feb 25 2020 4:20PM"
    dt_obj = string_to_datetime(date_str, '%b %d %Y %I:%M%p')
    print(f"String: '{date_str}' -> Datetime object: {dt_obj}")

    print("\n--- 20. Days Between Dates ---")
    date_1 = datetime(2020, 2, 25)
    date_2 = datetime(2020, 9, 17)
    days = days_between_dates(date_1, date_2)
    print(f"Days between {date_1.date()} and {date_2.date()}: {days}")

    # --- 7. Random Data Generation ---
    print("\n--- 21. Generate Random String ---")
    random_str = generate_random_string(10)
    print(f"Random 10-character string: {random_str}")
    
    print("\n--- 22. Generate Secure Random Integer ---")
    secure_int = generate_secure_random_int(100)
    print(f"Secure random integer below 100: {secure_int}")


if __name__ == "__main__":
    main()
