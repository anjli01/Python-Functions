# refactored_code.py

"""
This file contains a collection of refactored Python code examples.
The original code has been improved for readability, efficiency, and adherence
to standard Python (PEP 8) conventions.

Key improvements include:
- Descriptive variable and function names.
- Docstrings and type hints for clarity.
- Use of Pythonic idioms like list comprehensions.
- Replacement of inefficient algorithms with standard library functions.
- Encapsulation of logic into well-defined functions.
- A main execution block to demonstrate function usage.
"""

import math
import string
from datetime import datetime, timedelta
from functools import reduce
from itertools import permutations
from typing import Any, Dict, Generator, Iterable, List, Set, Tuple, Union

# ==============================================================================
# 1. File Operations
# ==============================================================================

def count_blank_spaces(file_path: str) -> int:
    """Counts the number of blank spaces in a text file.

    Args:
        file_path: The path to the text file.

    Returns:
        The total count of space characters (' ').
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            return content.count(' ')
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return 0

def capitalize_words_in_file(file_path: str) -> List[str]:
    """Capitalizes the first letter of every word in a file.

    Args:
        file_path: The path to the text file.

    Returns:
        A list of strings, where each string is a line with words capitalized.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [line.title() for line in f]
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []

def read_file_in_reverse(file_path: str) -> List[str]:
    """Reads the contents of a file and returns the lines in reverse order.

    Args:
        file_path: The path to the text file.

    Returns:
        A list of lines from the file in reverse order, with trailing whitespace stripped.
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            return [line.rstrip() for line in reversed(lines)]
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []


# ==============================================================================
# 2. List and Data Structure Operations
# ==============================================================================

def flatten_list(nested_list: Iterable) -> Generator[Any, None, None]:
    """Flattens a nested list of any depth into a single generator.

    Args:
        nested_list: A list that may contain other lists.

    Yields:
        Elements from the flattened list.
    """
    for item in nested_list:
        if isinstance(item, list):
            yield from flatten_list(item)
        else:
            yield item

def partition_even_odd(numbers: List[Union[int, float]]) -> Tuple[List, List]:
    """Separates elements of a list into two lists: one for even and one for odd.

    Args:
        numbers: A list of numbers.

    Returns:
        A tuple containing two lists: (even_numbers, odd_numbers).
    """
    even_numbers = [num for num in numbers if num % 2 == 0]
    odd_numbers = [num for num in numbers if num % 2 != 0]
    return even_numbers, odd_numbers

def sort_sublists_by_second_element(list_of_lists: List[List]) -> List[List]:
    """Sorts a list of sublists based on the second element of each sublist.

    Args:
        list_of_lists: The list to sort.

    Returns:
        A new list sorted by the second element of the sublists.
    """
    return sorted(list_of_lists, key=lambda x: x[1])

def find_second_largest(numbers: List[Union[int, float]]) -> Union[int, float, None]:
    """Finds the second largest number in a list.

    Args:
        numbers: A list of numbers.

    Returns:
        The second largest number, or None if the list has fewer than two unique elements.
    """
    unique_numbers = sorted(list(set(numbers)))
    if len(unique_numbers) < 2:
        return None
    return unique_numbers[-2]

def find_list_intersection(list1: List, list2: List) -> List:
    """Finds the intersection of two lists.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        A list containing elements common to both lists.
    """
    return list(set(list1) & set(list2))

def swap_first_and_last(data: List) -> List:
    """Swaps the first and last elements of a list.

    Args:
        data: The list to modify.

    Returns:
        The list with the first and last elements swapped.
    """
    if len(data) >= 2:
        data[0], data[-1] = data[-1], data[0]
    return data

def remove_duplicates(data: List) -> List:
    """Removes duplicate items from a list while preserving order.

    Args:
        data: The list with potential duplicates.

    Returns:
        A new list with duplicates removed.
    """
    seen = set()
    result = []
    for item in data:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

def find_longest_word(words: List[str]) -> Union[str, None]:
    """Finds the longest word in a list of words.

    Args:
        words: A list of strings.

    Returns:
        The longest word in the list, or None if the list is empty.
    """
    if not words:
        return None
    return max(words, key=len)

def remove_nth_occurrence(
    words: List[Any], item_to_remove: Any, n: int
) -> List[Any]:
    """Removes the nth occurrence of a given item from a list.

    Args:
        words: The list to process.
        item_to_remove: The item to look for.
        n: The occurrence to remove (e.g., 1 for the first, 2 for the second).

    Returns:
        A new list with the nth occurrence removed.
    """
    occurrence_count = 0
    result = []
    for item in words:
        if item == item_to_remove:
            occurrence_count += 1
            if occurrence_count != n:
                result.append(item)
        else:
            result.append(item)
    return result

def find_odd_occurring_element(numbers: List[int]) -> int:
    """Finds the element that occurs an odd number of times in a list.
    Assumes exactly one such element exists.

    Args:
        numbers: A list where one element appears an odd number of times,
                 and all others appear an even number of times.

    Returns:
        The element that occurs an odd number of times.
    """
    return reduce(lambda x, y: x ^ y, numbers)


# ==============================================================================
# 3. String Operations
# ==============================================================================

def count_vowels(text: str) -> int:
    """Counts the number of vowels in a string (case-insensitive).

    Args:
        text: The input string.

    Returns:
        The total count of vowels.
    """
    vowels = "aeiou"
    return sum(1 for char in text.lower() if char in vowels)

def find_common_letters(str1: str, str2: str) -> Set[str]:
    """Finds the common letters between two strings.

    Args:
        str1: The first string.
        str2: The second string.

    Returns:
        A set of letters common to both strings.
    """
    return set(str1) & set(str2)

def find_string_difference(str1: str, str2: str) -> Set[str]:
    """Finds letters that are in the first string but not in the second.

    Args:
        str1: The first string.
        str2: The second string (the set to subtract).

    Returns:
        A set of letters present in str1 but not in str2.
    """
    return set(str1) - set(str2)

def remove_char_at_index(text: str, n: int) -> str:
    """Removes the character at the nth index from a string.

    Args:
        text: The input string.
        n: The index of the character to remove.

    Returns:
        A new string with the character removed.
    """
    if 0 <= n < len(text):
        return text[:n] + text[n + 1 :]
    return text

def are_anagrams(str1: str, str2: str) -> bool:
    """Checks if two strings are anagrams of each other.

    Args:
        str1: The first string.
        str2: The second string.

    Returns:
        True if the strings are anagrams, False otherwise.
    """
    return sorted(str1.lower().replace(" ", "")) == sorted(str2.lower().replace(" ", ""))

def exchange_first_last_char(text: str) -> str:
    """Forms a new string where the first and last characters have been exchanged.

    Args:
        text: The input string.

    Returns:
        A new string with first and last characters swapped.
    """
    if len(text) < 2:
        return text
    return text[-1] + text[1:-1] + text[0]

def get_even_index_chars(text: str) -> str:
    """Creates a new string containing only characters from even indices.

    Args:
        text: The input string.

    Returns:
        A new string made of characters from even indices.
    """
    return text[::2]

def find_larger_string(str1: str, str2: str) -> Union[str, None]:
    """Compares two strings and returns the one with the greater length.

    Args:
        str1: The first string.
        str2: The second string.

    Returns:
        The longer string, or None if they have equal length.
    """
    if len(str1) > len(str2):
        return str1
    if len(str2) > len(str1):
        return str2
    return None

def count_lowercase_chars(text: str) -> int:
    """Counts the number of lowercase characters in a string.

    Args:
        text: The input string.

    Returns:
        The total count of lowercase ASCII characters.
    """
    return sum(1 for char in text if char.islower())


# ==============================================================================
# 4. Dictionary Operations
# ==============================================================================

def merge_dictionaries(dict1: Dict, dict2: Dict) -> Dict:
    """Merges two dictionaries into a new dictionary.
    Keys from dict2 will overwrite keys from dict1 in case of conflicts.

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.

    Returns:
        A new dictionary containing items from both.
    """
    return {**dict1, **dict2}

def multiply_dict_values(data: Dict[Any, Union[int, float]]) -> Union[int, float]:
    """Multiplies all the numeric values in a dictionary.

    Args:
        data: A dictionary with numeric values.

    Returns:
        The product of all values.
    """
    return reduce(lambda x, y: x * y, data.values(), 1)

def remove_dict_key(data: Dict, key_to_remove: Any) -> bool:
    """Removes a given key from a dictionary in-place.

    Args:
        data: The dictionary to modify.
        key_to_remove: The key to remove.

    Returns:
        True if the key was found and removed, False otherwise.
    """
    if key_to_remove in data:
        del data[key_to_remove]
        return True
    print(f"Key '{key_to_remove}' not found!")
    return False

def map_lists_to_dict(keys: List, values: List) -> Dict:
    """Maps two lists into a dictionary.

    Args:
        keys: A list of keys.
        values: A list of values.

    Returns:
        A dictionary created from the keys and values.
    """
    return dict(zip(keys, values))


# ==============================================================================
# 5. Mathematical and Algorithmic Functions
# ==============================================================================

def gcd(a: int, b: int) -> int:
    """Computes the Greatest Common Divisor (GCD) of two numbers using Euclidean algorithm.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The GCD of a and b.
    """
    while b:
        a, b = b, a % b
    return a

def lcm(a: int, b: int) -> int:
    """Computes the Least Common Multiple (LCM) of two numbers.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        The LCM of a and b.
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // gcd(a, b)

def is_prime(n: int) -> bool:
    """Checks if a number is a prime number.

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

def power(base: Union[int, float], exp: int) -> Union[int, float]:
    """Calculates the power of a number using recursion.

    Args:
        base: The base number.
        exp: The exponent (must be a non-negative integer).

    Returns:
        The result of base raised to the power of exp.
    """
    if exp < 0:
        raise ValueError("Exponent must be a non-negative integer.")
    if exp == 0:
        return 1
    if exp == 1:
        return base
    return base * power(base, exp - 1)

def sum_nested_list(nested_list: List) -> Union[int, float]:
    """Calculates the total sum of all numbers in a possibly nested list.

    Args:
        nested_list: A list that may contain numbers and other lists of numbers.

    Returns:
        The sum of all numeric elements.
    """
    total = 0
    for element in nested_list:
        if isinstance(element, list):
            total += sum_nested_list(element)
        elif isinstance(element, (int, float)):
            total += element
    return total

def is_leap_year(year: int) -> bool:
    """Checks if a given year is a leap year.

    Args:
        year: The year to check.

    Returns:
        True if it is a leap year, False otherwise.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def get_prime_factors(n: int) -> List[int]:
    """Finds all prime factors of an integer.

    Args:
        n: The integer to factorize.

    Returns:
        A list of prime factors.
    """
    factors = []
    d = 2
    while d * d <= n:
        while (n % d) == 0:
            factors.append(d)
            n //= d
        d += 1
    if n > 1:
       factors.append(n)
    return factors

def get_divisors(n: int) -> List[int]:
    """Finds all divisors of an integer.

    Args:
        n: The integer.

    Returns:
        A list of divisors.
    """
    if n <= 0:
        return []
    return [i for i in range(1, n + 1) if n % i == 0]

def is_armstrong_number(n: int) -> bool:
    """Checks if a number is an Armstrong number.
    An Armstrong number is a number that is equal to the sum of its own digits
    each raised to the power of the number of digits.
    """
    if n < 0:
        return False
    s = str(n)
    num_digits = len(s)
    return n == sum(int(digit) ** num_digits for digit in s)

def generate_pascals_triangle(rows: int) -> List[List[int]]:
    """Generates Pascal's triangle for a given number of rows.

    Args:
        rows: The number of rows to generate.

    Returns:
        A list of lists representing the triangle.
    """
    if rows <= 0:
        return []
    triangle = [[1]]
    for i in range(1, rows):
        prev_row = triangle[-1]
        new_row = [1]
        for j in range(len(prev_row) - 1):
            new_row.append(prev_row[j] + prev_row[j+1])
        new_row.append(1)
        triangle.append(new_row)
    return triangle

def factorial(n: int) -> int:
    """Calculates the factorial of a number."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    return math.factorial(n)

def is_strong_number(n: int) -> bool:
    """Checks if a number is a Strong Number.
    A strong number is a number where the sum of the factorial of its digits is
    equal to the number itself.
    """
    if n < 0:
        return False
    return n == sum(factorial(int(digit)) for digit in str(n))

def find_pythagorean_triplets(limit: int) -> List[Tuple[int, int, int]]:
    """Generates Pythagorean triplets (a, b, c) such that a^2 + b^2 = c^2 and c <= limit.

    Args:
        limit: The maximum value for c.

    Returns:
        A list of tuples, where each tuple is a Pythagorean triplet.
    """
    triplets = []
    for a in range(1, limit + 1):
        for b in range(a, limit + 1):
            c_square = a**2 + b**2
            c = int(math.sqrt(c_square))
            if c_square == c**2 and c <= limit:
                triplets.append((a, b, c))
    return triplets

def print_collatz_sequence(n: int):
    """Prints the Collatz sequence for a given starting number.

    Args:
        n: A positive integer to start the sequence from.
    """
    if n <= 0:
        print("Please provide a positive integer.")
        return
    while n > 1:
        print(n, end=' -> ')
        if n % 2:  # n is odd
            n = 3 * n + 1
        else:  # n is even
            n = n // 2
    print(1)

def is_palindrome(text: str) -> bool:
    """Checks if a string is a palindrome (reads the same forwards and backward).
    This check is case-insensitive.
    """
    normalized_text = text.casefold()
    return normalized_text == normalized_text[::-1]

def reverse_words_in_sentence(sentence: str) -> str:
    """Reverses the order of words in a sentence."""
    words = sentence.split()
    return " ".join(reversed(words))


# ==============================================================================
# 6. Date and Time Functions
# ==============================================================================

def get_next_date(date_str: str) -> Union[str, None]:
    """Validates a date string (DD/MM/YYYY) and returns the next day's date.

    Args:
        date_str: The date string in "DD/MM/YYYY" format.

    Returns:
        The incremented date as a string in the same format, or None if the
        input date is invalid.
    """
    try:
        current_date = datetime.strptime(date_str, "%d/%m/%Y")
        next_day = current_date + timedelta(days=1)
        return next_day.strftime("%d/%m/%Y")
    except ValueError:
        print(f"'{date_str}' is an invalid date.")
        return None


# ==============================================================================
# 7. Binary and Bitwise Operations
# ==============================================================================

def count_set_bits(n: int) -> int:
    """Counts the number of set bits (1s) in the binary representation of an integer.

    Args:
        n: A non-negative integer.

    Returns:
        The count of set bits.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    return bin(n).count('1')

def generate_gray_codes(n_bits: int) -> List[str]:
    """Generates n-bit Gray codes recursively.

    Args:
        n_bits: The number of bits for the Gray code.

    Returns:
        A list of strings representing the n-bit Gray codes.
    """
    if n_bits == 0:
        return ['']
    if n_bits == 1:
        return ['0', '1']

    prev_codes = generate_gray_codes(n_bits - 1)
    
    first_half = ['0' + code for code in prev_codes]
    second_half = ['1' + code for code in reversed(prev_codes)]

    return first_half + second_half

def gray_to_binary(gray_code: str) -> str:
    """Converts a Gray code string to its binary equivalent string."""
    binary = gray_code[0]
    for i in range(1, len(gray_code)):
        if gray_code[i] == '0':
            binary += binary[i-1]
        else:
            binary += '1' if binary[i-1] == '0' else '0'
    return binary

def binary_to_gray(binary_str: str) -> str:
    """Converts a binary string to its Gray code equivalent string."""
    gray = binary_str[0]
    for i in range(1, len(binary_str)):
        gray += str(int(binary_str[i-1]) ^ int(binary_str[i]))
    return gray


# ==============================================================================
# 8. Main Execution Block (Demonstrations)
# ==============================================================================

if __name__ == "__main__":
    print("--- Running Demonstrations ---")

    # 1. File Operations (requires a 'sample.txt' file)
    print("\n--- File Operations ---")
    try:
        with open("sample.txt", "w", encoding="utf-8") as f:
            f.write("hello world\npython is fun\n")
        print(f"Blank spaces in sample.txt: {count_blank_spaces('sample.txt')}")
        capitalized_lines = capitalize_words_in_file('sample.txt')
        print(f"Capitalized content: {capitalized_lines}")
        reversed_lines = read_file_in_reverse('sample.txt')
        print(f"Reversed content: {reversed_lines}")
    except IOError as e:
        print(f"Could not write to sample.txt: {e}")

    # 2. List Operations
    print("\n--- List Operations ---")
    nested = [[1, [[2]], [[[3]]]], [[4], 5]]
    flat = list(flatten_list(nested))
    print(f"Flattening {nested}: {flat}")
    numbers_to_partition = [2, 3, 8, 9, 2, 4, 6]
    evens, odds = partition_even_odd(numbers_to_partition)
    print(f"Partitioning {numbers_to_partition}: Evens={evens}, Odds={odds}")
    sublists = [['A', 34], ['B', 21], ['C', 26]]
    print(f"Sorting {sublists} by 2nd element: {sort_sublists_by_second_element(sublists)}")
    print(f"Second largest in {numbers_to_partition}: {find_second_largest(numbers_to_partition)}")
    list1, list2 = [1, 2, 3, 4], [3, 4, 5, 6]
    print(f"Intersection of {list1} and {list2}: {find_list_intersection(list1, list2)}")
    my_list = [10, 20, 30, 40]
    print(f"Swapping first/last in {my_list}: {swap_first_and_last(my_list.copy())}")

    # 3. String Operations
    print("\n--- String Operations ---")
    my_string = "This is an Assignment"
    print(f"Vowels in '{my_string}': {count_vowels(my_string)}")
    print(f"Lowercase chars in '{my_string}': {count_lowercase_chars(my_string)}")
    str1, str2 = 'python', 'schoolofai'
    print(f"Common letters between '{str1}' and '{str2}': {find_common_letters(str1, str2)}")
    print(f"Are '{str1}' and 'thonpy' anagrams? {are_anagrams(str1, 'thonpy')}")

    # 4. Dictionary Operations
    print("\n--- Dictionary Operations ---")
    dict1, dict2 = {'a': 1, 'b': 2}, {'b': 3, 'c': 4}
    print(f"Merging {dict1} and {dict2}: {merge_dictionaries(dict1, dict2)}")
    num_dict = {'x': 2, 'y': 5, 'z': 10}
    print(f"Product of values in {num_dict}: {multiply_dict_values(num_dict)}")

    # 5. Math and Algorithms
    print("\n--- Math and Algorithms ---")
    print(f"GCD of 48 and 18: {gcd(48, 18)}")
    print(f"LCM of 4 and 7: {lcm(4, 7)}")
    print(f"Is 29 a prime number? {is_prime(29)}")
    print(f"Power of 3^4: {power(3, 4)}")
    print(f"Sum of nested list {nested}: {sum_nested_list(nested)}")
    print(f"Is 2024 a leap year? {is_leap_year(2024)}")
    print(f"Is 153 an Armstrong number? {is_armstrong_number(153)}")
    print("Pascal's Triangle (5 rows):")
    for row in generate_pascals_triangle(5):
        print(row)
    print("Collatz sequence for 10:")
    print_collatz_sequence(10)

    # 6. Date Functions
    print("\n--- Date Functions ---")
    print(f"Day after 31/12/2023: {get_next_date('31/12/2023')}")
    print(f"Day after 28/02/2023: {get_next_date('28/02/2023')}")
    print(f"Trying an invalid date: {get_next_date('32/01/2023')}")

    print("\n--- Refactoring Complete ---")