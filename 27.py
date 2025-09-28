"""
A collection of utility functions for string manipulation.
"""
import re
from collections import Counter
from typing import List, Set, Tuple

def get_alternating_case(text: str) -> str:
    """Converts a string to alternating upper and lower case."""
    return "".join(
        char.upper() if i % 2 == 0 else char.lower()
        for i, char in enumerate(text)
    )

def is_palindrome(text: str) -> bool:
    """Checks if a string is a palindrome."""
    return text == text[::-1]

def is_symmetrical(text: str) -> bool:
    """Checks if a string's first half is identical to its second half."""
    n = len(text)
    mid = n // 2
    first_half = text[:mid]
    # Handle odd length by excluding the middle character
    second_half = text[mid:] if n % 2 == 0 else text[mid + 1:]
    return first_half == second_half

def reverse_words(sentence: str) -> str:
    """Reverses the order of words in a sentence."""
    words = sentence.split()
    return ' '.join(reversed(words))

def is_substring_present(main_string: str, substring: str) -> bool:
    """Checks if a substring is present in a string."""
    return substring in main_string

def get_word_frequency(text: str) -> Counter:
    """Calculates the frequency of each word in a string."""
    return Counter(text.split())

def find_even_length_words(text: str) -> List[str]:
    """Finds all words with an even length in a string."""
    return [word for word in text.split() if len(word) % 2 == 0]

def contains_all_vowels(text: str) -> bool:
    """Checks if a string contains all vowels (case-insensitive)."""
    vowels = "aeiou"
    return set(vowels).issubset(set(text.lower()))

def count_unique_matching_chars(str1: str, str2: str) -> int:
    """Counts the number of unique characters common to two strings."""
    return len(set(str1) & set(str2))

def remove_duplicate_chars(text: str) -> str:
    """Removes duplicate characters from a string, preserving order."""
    return "".join(dict.fromkeys(text))

def get_char_frequency(text: str) -> Counter:
    """Returns a Counter object with the frequency of each character."""
    return Counter(text)

def find_words_shorter_than(text: str, max_length: int) -> List[str]:
    """Finds all words in a string shorter than a given length."""
    return [word for word in text.split() if len(word) < max_length]

def is_binary_string(text: str) -> bool:
    """Checks if a string contains only '0's and '1's."""
    return all(char in '01' for char in text)

def remove_char_at_index(text: str, index: int) -> str:
    """Removes a character from a string at a specific index."""
    if 0 <= index < len(text):
        return text[:index] + text[index + 1:]
    return text

def find_urls(text: str) -> List[str]:
    """Finds all URLs in a string using regex."""
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    urls_found = re.findall(regex, text)
    return [url[0] for url in urls_found]

def find_uncommon_words(str1: str, str2: str) -> List[str]:
    """Finds words that appear in exactly one of the two strings."""
    count = Counter(str1.split()) + Counter(str2.split())
    return [word for word, freq in count.items() if freq == 1]

def rotate_string(text: str, positions: int, direction: str = 'left') -> str:
    """Rotates a string left or right by a given number of positions."""
    if not text:
        return ""
    positions %= len(text)
    if direction == 'left':
        return text[positions:] + text[:positions]
    elif direction == 'right':
        return text[-positions:] + text[:-positions]
    return text

def move_digits_to_end(text: str) -> str:
    """Moves all numeric digits to the end of the string."""
    chars = "".join(c for c in text if not c.isdigit())
    digits = "".join(c for c in text if c.isdigit())
    return chars + digits

def count_surrounding_chars(text: str, chars_to_check: Set[str]) -> int:
    """Counts non-check characters that are adjacent to a check character."""
    if len(text) < 2:
        return 0
    
    count = 0
    for i, char in enumerate(text):
        if char in chars_to_check:
            continue
        # Check left neighbor
        if i > 0 and text[i - 1] in chars_to_check:
            count += 1
            continue # Count only once per character
        # Check right neighbor
        if i < len(text) - 1 and text[i + 1] in chars_to_check:
            count += 1
    return count

def count_spaces(text: str) -> int:
    """Counts the number of space characters in a string."""
    return text.count(' ')

def find_substrings_with_k_distinct(text: str, n: int, k: int) -> List[str]:
    """Extracts substrings of length N with K distinct characters."""
    result = []
    for i in range(len(text) - n + 1):
        substring = text[i:i + n]
        if len(set(substring)) == k:
            result.append(substring)
    return result

def increment_trailing_number(text: str) -> str:
    """Finds a trailing number in a string and increments it, preserving padding."""
    match = re.search(r'(\d+)$', text)
    if not match:
        return text
    
    number_str = match.group(1)
    number_len = len(number_str)
    incremented_num = int(number_str) + 1
    
    # zfill pads with zeros
    new_number_str = str(incremented_num).zfill(number_len)
    
    return text[:match.start(1)] + new_number_str

def count_letters_and_digits(text: str) -> Tuple[int, int]:
    """Counts the total number of letters and digits in a string."""
    letters = sum(1 for char in text if char.isalpha())
    digits = sum(1 for char in text if char.isdigit())
    return letters, digits

def has_lowercase(text: str) -> bool:
    """Checks if any character in the string is lowercase."""
    return any(c.islower() for c in text)

def has_uppercase(text: str) -> bool:
    """Checks if any character in the string is uppercase."""
    return any(c.isupper() for c in text)



"""
A collection of utility functions for list and tuple manipulation.
"""
from typing import List, Any, Tuple, TypeVar, Union

T = TypeVar('T')

def are_all_elements_identical(data: List[Any]) -> bool:
    """Checks if all elements in a list are identical."""
    if not data:
        return True
    return len(set(data)) == 1

def find_positive_numbers(numbers: List[Union[int, float]]) -> List[Union[int, float]]:
    """Filters a list to return only the positive numbers."""
    return [num for num in numbers if num >= 0]

def multiply_elements(numbers: List[Union[int, float]]) -> Union[int, float]:
    """Multiplies all the numbers in a list."""
    if not numbers:
        return 1
    result = 1
    for num in numbers:
        result *= num
    return result

def remove_even_numbers(numbers: List[int]) -> List[int]:
    """Removes all even numbers from a list."""
    return [num for num in numbers if num % 2 != 0]

def remove_empty_strings(strings: List[str]) -> List[str]:
    """Removes all empty or whitespace-only strings from a list."""
    return [s for s in strings if s and not s.isspace()]

def get_cumulative_sum(numbers: List[Union[int, float]]) -> List[Union[int, float]]:
    """Calculates the cumulative sum of a list of numbers."""
    cumulative_list = []
    current_sum = 0
    for num in numbers:
        current_sum += num
        cumulative_list.append(current_sum)
    return cumulative_list

def distance_between_first_last_even(numbers: List[int]) -> int:
    """Calculates the index distance between the first and last even number."""
    indices = [i for i, num in enumerate(numbers) if num % 2 == 0]
    if len(indices) < 2:
        return 0
    return indices[-1] - indices[0]

def swap_first_last(data: List[T]) -> List[T]:
    """Swaps the first and last elements of a list."""
    if len(data) >= 2:
        data[0], data[-1] = data[-1], data[0]
    return data

def find_elements_containing_digit(numbers: List[int], digit: int) -> List[int]:
    """Finds all numbers in a list that contain a specific digit."""
    return [num for num in numbers if str(digit) in str(num)]

def count_unique_elements(data: List[Any]) -> int:
    """Counts the number of unique elements in a list."""
    return len(set(data))

def get_sum_and_average(numbers: List[Union[int, float]]) -> Tuple[float, float]:
    """Calculates the sum and average of a list of numbers."""
    if not numbers:
        return 0, 0
    total_sum = sum(numbers)
    average = total_sum / len(numbers)
    return total_sum, average

def remove_tuples_of_length(tuples: List[Tuple], length: int) -> List[Tuple]:
    """Removes all tuples of a specific length from a list of tuples."""
    return [t for t in tuples if len(t) != length]

def create_number_and_cube_tuples(numbers: List[int]) -> List[Tuple[int, int]]:
    """Creates a list of tuples, where each tuple is a number and its cube."""
    return [(num, num ** 3) for num in numbers]

def swap_tuple_elements(tuples: List[Tuple[T, T]]) -> List[Tuple[T, T]]:
    """Swaps the two elements in each tuple within a list."""
    return [(y, x) for x, y in tuples]

def sort_tuples_by_second_item(tuples: List[Tuple]) -> List[Tuple]:
    """Sorts a list of tuples based on the second element of each tuple."""
    return sorted(tuples, key=lambda x: x[1])

def get_all_pair_combinations(tuple1: Tuple, tuple2: Tuple) -> List[Tuple]:
    """Generates all pair combinations from two tuples."""
    pairs = [(a, b) for a in tuple1 for b in tuple2]
    pairs.extend([(b, a) for b in tuple2 for a in tuple1])
    return pairs



"""
A collection of utility functions for dictionary manipulation.
"""
from typing import List, Dict, Any, Hashable, Tuple

def lists_to_dict(keys: List[Hashable], values: List[Any]) -> Dict[Hashable, Any]:
    """Converts two lists of equal length into a dictionary."""
    return dict(zip(keys, values))

def sort_list_of_dicts(
    dict_list: List[Dict[str, Any]], sort_key: str
) -> List[Dict[str, Any]]:
    """Sorts a list of dictionaries by a specified key."""
    return sorted(dict_list, key=lambda item: item.get(sort_key))

def find_longest_key(data: Dict[str, Any]) -> str:
    """Finds the key with the maximum length in a dictionary."""
    if not data:
        return ""
    return max(data.keys(), key=len)

def capitalize_key_ends(data: Dict[str, Any]) -> Dict[str, Any]:
    """Capitalizes the first and last character of each key in a dictionary."""
    new_dict = {}
    for key, value in data.items():
        if len(key) >= 2:
            new_key = key[0].upper() + key[1:-1] + key[-1].upper()
            new_dict[new_key] = value
        elif key:
            new_dict[key.upper()] = value
    return new_dict

def invert_dict_unique_values(data: Dict[Hashable, Hashable]) -> Dict[Hashable, Hashable]:
    """Inverts a dictionary with unique values."""
    return {value: key for key, value in data.items()}

def invert_dict_non_unique(data: Dict[Hashable, Any]) -> Dict[Any, List[Hashable]]:
    """Inverts a dictionary, grouping keys with the same value into a list."""
    inverted = {}
    for key, value in data.items():
        inverted.setdefault(value, []).append(key)
    return inverted

def merge_list_of_dicts(dict_list: List[Dict]) -> Dict:
    """Merges a list of dictionaries into a single dictionary."""
    return {key: value for d in dict_list for key, value in d.items()}

def count_unique_keys_in_list_of_dicts(dict_list: List[Dict]) -> int:
    """Counts the number of unique keys across all dictionaries in a list."""
    return len(set(key for d in dict_list for key in d))

def replace_list_value_with_nth_element(
    dict_list: List[Dict], key: Hashable, n: int
) -> List[Dict]:
    """
    In a list of dicts, if a value for a key is a list, replaces it
    with its Nth element.
    """
    for sub_dict in dict_list:
        if key in sub_dict and isinstance(sub_dict[key], list) and len(sub_dict[key]) > n:
            sub_dict[key] = sub_dict[key][n]
    return dict_list

def get_index_wise_product_of_tuple_values(data: Dict[Any, Tuple]) -> Tuple:
    """Calculates the product of tuple elements at each index across all values."""
    if not data:
        return ()
    
    # zip(*data.values()) transposes the tuples
    # e.g., ((5,6,1), (8,3,2)) -> ((5,8), (6,3), (1,2))
    products = []
    for zipped_values in zip(*data.values()):
        product = 1
        for val in zipped_values:
            product *= val
        products.append(product)
    return tuple(products)

def sort_nested_dict_by_key(data: Dict[Any, Dict], sort_key: str) -> Dict:
    """Sorts a nested dictionary by a key in the inner dictionary."""
    return dict(sorted(data.items(), key=lambda item: item[1][sort_key]))

def combine_lists_to_nested_dict(
    list1: List, list2: List, list3: List
) -> List[Dict]:
    """Combines three lists into a list of nested dictionaries."""
    return [{a: {b: c}} for a, b, c in zip(list1, list2, list3)]

def flatten_dict(
    nested_dict: Dict, separator: str = '_', prefix: str = ''
) -> Dict:
    """Flattens a nested dictionary into a single-level dictionary."""
    flat_dict = {}
    for k, v in nested_dict.items():
        new_key = f"{prefix}{separator}{k}" if prefix else k
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, separator, new_key))
        else:
            flat_dict[new_key] = v
    return flat_dict

def get_top_n_largest_keys(data: Dict[int, Any], n: int) -> List[int]:
    """Returns the top N largest integer keys from a dictionary."""
    return sorted(data.keys(), reverse=True)[:n]



"""
This script demonstrates the usage of the refactored utility functions
from the string_operations, list_operations, and dictionary_operations modules.
"""

import sys
import datetime
import time

# Import all functions from the custom modules
import string_operations as so
import list_operations as lo
import dictionary_operations as do

def main():
    """Main function to run all examples."""
    print("--- Running String Operations Examples ---")
    
    # Get alternating case
    s = "geeksforgeeks"
    print(f"Original: '{s}', Alternating Case: '{so.get_alternating_case(s)}'")

    # Palindrome and Symmetrical checks
    s_palindrome = "madam"
    s_symmetrical = "abccba"
    print(f"'{s_palindrome}' is palindrome: {so.is_palindrome(s_palindrome)}")
    print(f"'{s_symmetrical}' is symmetrical: {so.is_symmetrical(s_symmetrical)}")
    
    # Reverse words in a sentence
    sentence = "python is fun"
    print(f"Original: '{sentence}', Reversed: '{so.reverse_words(sentence)}'")

    # Word frequency
    text = "It is a great meal at a great restaurant"
    print(f"Word frequency in '{text}': {so.get_word_frequency(text)}")
    
    # Find uncommon words
    s1, s2 = "apple banana orange", "apple mango banana"
    print(f"Uncommon words between '{s1}' and '{s2}': {so.find_uncommon_words(s1, s2)}")
    
    # Rotate string
    s = "HelloWorld"
    print(f"Rotate '{s}' left by 2: {so.rotate_string(s, 2, 'left')}")
    print(f"Rotate '{s}' right by 3: {so.rotate_string(s, 3, 'right')}")
    
    # Increment trailing number
    s_num = "file_version_009"
    print(f"Incrementing '{s_num}': {so.increment_trailing_number(s_num)}")

    print("\n--- Running List and Tuple Operations Examples ---")

    # Get memory size of a list
    my_list = ['Scott', 'Eric', 'Kelly', 'Emma', 'Smith']
    print(f"Memory size of list {my_list} = {sys.getsizeof(my_list)} bytes")

    # Check if all elements are identical
    list_one = [20, 20, 20, 20]
    list_two = [20, 21, 20, 20]
    print(f"All elements identical in {list_one}: {lo.are_all_elements_identical(list_one)}")
    print(f"All elements identical in {list_two}: {lo.are_all_elements_identical(list_two)}")

    # Multiply elements
    numbers = [2, 3, 4]
    print(f"Product of {numbers} is {lo.multiply_elements(numbers)}")

    # Remove even numbers
    numbers = [11, 5, 17, 18, 23, 50]
    print(f"Original: {numbers}, After removing evens: {lo.remove_even_numbers(numbers)}")

    # Cumulative sum
    numbers = [10, 20, 30, 40, 50]
    print(f"Cumulative sum of {numbers}: {lo.get_cumulative_sum(numbers)}")

    # Swap first and last elements
    data = [12, 35, 9, 56, 24]
    print(f"Original: {data}, Swapped: {lo.swap_first_last(data.copy())}")
    
    # Sort tuples by second item
    tuples = [('a', 3), ('b', 1), ('c', 2)]
    print(f"Original tuples: {tuples}, Sorted by 2nd item: {lo.sort_tuples_by_second_item(tuples)}")


    print("\n--- Running Dictionary Operations Examples ---")

    # Merge two dictionaries
    current_emp = {1: 'Scott', 2: "Eric", 3:"Kelly"}
    former_emp  = {2: 'Eric', 4: "Emma"}
    all_emp = {**current_emp, **former_emp}
    print(f"Merged dictionary: {all_emp}")

    # Convert two lists to a dictionary
    item_ids = [54, 65, 76]
    names = ["Hard Disk", "Laptop", "RAM"]
    item_dict = do.lists_to_dict(item_ids, names)
    print(f"Dictionary from lists: {item_dict}")
    
    # Find longest key
    d = {"key1": 10, "super_long_key": 2, "k3": 30}
    print(f"Longest key in {d}: '{do.find_longest_key(d)}'")
    
    # Invert dictionary (non-unique values)
    d_non_unique = {'a': 1, 'b': 2, 'c': 1, 'd': 3}
    print(f"Inverted dict for {d_non_unique}: {do.invert_dict_non_unique(d_non_unique)}")
    
    # Flatten a nested dictionary
    nested = {'user': {'name': 'Alice', 'address': {'city': 'New York'}}}
    print(f"Flattened dictionary: {do.flatten_dict(nested)}")
    
    # Date to timestamp conversion
    date_str = "20/01/2020"
    element = datetime.datetime.strptime(date_str, "%d/%m/%Y")
    timestamp = time.mktime(element.timetuple())
    print(f"Timestamp for {date_str} is: {timestamp}")

if __name__ == "__main__":
    main()
