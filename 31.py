# ==============================================================================
# IMPORTS
# ==============================================================================
import calendar
import keyword
import math
import os
import re
from datetime import datetime, timedelta

# Third-party libraries (if any, require installation)
# To install: pip install numpy pandas psutil
try:
    import numpy as np
    import pandas as pd
    import psutil
except ImportError:
    print("Please install required libraries: pip install numpy pandas psutil")
    # You can decide to exit here or handle the missing libraries gracefully.
    np = pd = psutil = None

# ==============================================================================
# REFACTORED FUNCTIONS
# ==============================================================================


# write a Python Program to check if a number is a Perfect number and print the result
n = 7
sum1 = 0
for i in range(1, n):
    if(n % i == 0):
        sum1 = sum1 + i
if (sum1 == n):
    print("The number is a Perfect number!")
else:
    print("The number is not a Perfect number!")


def is_perfect(number: int) -> bool:
    """Checks if a number is a Perfect Number.

    A perfect number is a positive integer that is equal to the sum of its
    proper positive divisors (the sum of its positive divisors excluding
    the number itself).

    Args:
        number: The integer to check.

    Returns:
        True if the number is a perfect number, False otherwise.
    """
    if number <= 0:
        return False
    
    # Sum of divisors from 1 up to number // 2
    sum_of_divisors = sum(i for i in range(1, number // 2 + 1) if number % i == 0)
    
    return sum_of_divisors == number


# write a Python function to convert binary to Gray codeword
def binary_to_gray(n):
    n = int(n, 2)
    n ^= (n >> 1)
    return bin(n)[2:]
 
# write a Python function to convert Gray code to binary 
def gray_to_binary(n):
    n = int(n, 2) # convert to int
    mask = n
    while mask != 0:
        mask >>= 1
        n ^= mask
    return bin(n)[2:]



def binary_to_gray(binary_str: str) -> str:
    """Converts a binary string to a Gray codeword string.

    Args:
        binary_str: The input binary string (e.g., "1010").

    Returns:
        The corresponding Gray codeword as a string.
    """
    num = int(binary_str, 2)
    gray_code = num ^ (num >> 1)
    return bin(gray_code)[2:]

def gray_to_binary(gray_str: str) -> str:
    """Converts a Gray codeword string to a binary string.

    Args:
        gray_str: The input Gray codeword string (e.g., "1111").

    Returns:
        The corresponding binary string.
    """
    num = int(gray_str, 2)
    mask = num
    while mask != 0:
        mask >>= 1
        num ^= mask
    return bin(num)[2:]


# write a Python Program to Replace all Occurrences of ‘a’ with $ in a String
def replacestring(txt):
	return txt.replace('A','$')



def replace_char_in_string(text: str, old_char: str, new_char: str) -> str:
    """Replaces all occurrences of a character in a string with another.

    Args:
        text: The original string.
        old_char: The character to be replaced.
        new_char: The character to replace with.

    Returns:
        The modified string.
    """
    return text.replace(old_char, new_char)



# write a python program to Print Quotient and Remainder of two numbers 
a = 15
b = 4
quotient=a//b
remainder=a%b
print("Quotient is:",quotient)
print("Remainder is:",remainder)



def get_quotient_and_remainder(dividend: int, divisor: int) -> tuple[int, int]:
    """Calculates the quotient and remainder of two numbers.

    Args:
        dividend: The number to be divided.
        divisor: The number to divide by.

    Returns:
        A tuple containing the quotient and the remainder.
    """
    if divisor == 0:
        raise ValueError("Divisor cannot be zero.")
    return divmod(dividend, divisor)


# write a python program to print the Area of a Triangle Given All Three Sides
a = 15
b = 9
c = 7
s=(a+b+c)/2
area=(s*(s-a)*(s-b)*(s-c)) ** 0.5
print("Area of the triangle is: ",round(area,2))



def calculate_triangle_area(side_a: float, side_b: float, side_c: float) -> float | None:
    """Calculates the area of a triangle using Heron's formula.

    Args:
        side_a: Length of the first side.
        side_b: Length of the second side.
        side_c: Length of the third side.

    Returns:
        The area of the triangle as a float, or None if the sides
        do not form a valid triangle.
    """
    # Check for triangle inequality
    if not (side_a + side_b > side_c and 
            side_a + side_c > side_b and 
            side_b + side_c > side_a):
        return None

    s = (side_a + side_b + side_c) / 2
    area = math.sqrt(s * (s - side_a) * (s - side_b) * (s - side_c))
    return area



# write a Python function to Determine all Pythagorean Triplets in the Range
def findpythagoreantriplets(limit):
	c=0
	m=2
	while(c<limit):
		for n in range(1,m+1):
			a=m*m-n*n
			b=2*m*n
			c=m*m+n*n
			if(c>limit):
				break
			if(a==0 or b==0 or c==0):
				break
			print(a,b,c)
		m=m+1


from collections.abc import Generator

def find_pythagorean_triplets(limit: int) -> Generator[tuple[int, int, int], None, None]:
    """Generates Pythagorean triplets (a, b, c) where c is up to a given limit.
    
    Uses Euclid's formula: a = m^2 - n^2, b = 2mn, c = m^2 + n^2.

    Args:
        limit: The maximum value for c.

    Yields:
        A tuple (a, b, c) representing a Pythagorean triplet.
    """
    m = 2
    while True:
        c_test = m * m + 1  # Smallest c for a given m is with n=1
        if c_test > limit:
            break
            
        for n in range(1, m):
            # To generate primitive triplets, m and n must be coprime
            # and one of them must be even. We can add this for efficiency later.
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n
            
            if c > limit:
                break

            # Ensure a is the smaller leg
            yield tuple(sorted((a, b)) + [c])
        
        m += 1


"""
A collection of simple Python code examples, refactored for clarity,
reusability, and adherence to standard practices (PEP 8).

Each piece of original procedural code has been encapsulated into a
well-documented function with type hints. A demonstration of how to
use each function is provided in the main execution block at the end.
"""

# ==============================================================================
# IMPORTS
# ==============================================================================
import calendar
import keyword
import math
import os
import re
from collections.abc import Generator, Iterable, Sequence
from datetime import datetime, timedelta
from typing import Any, TypeVar

# Third-party libraries (require installation: pip install numpy pandas psutil)
try:
    import numpy as np
    import pandas as pd
    import psutil
except ImportError:
    print("Warning: numpy, pandas, or psutil not found. Some functions will not work.")
    print("Please install with: pip install numpy pandas psutil")
    np = pd = psutil = None

# ==============================================================================
# TYPE ALIASES for better readability
# ==============================================================================
T = TypeVar('T')
Matrix = list[list[float | int]]


# ==============================================================================
# REFACTORED FUNCTIONS
# ==============================================================================

# --- Number Theory & Math ---

def is_perfect(number: int) -> bool:
    """Checks if a number is a Perfect Number.

    A perfect number is a positive integer that is equal to the sum of its
    proper positive divisors (the sum of its positive divisors excluding
    the number itself).

    Args:
        number: The integer to check.

    Returns:
        True if the number is a perfect number, False otherwise.
    """
    if number <= 1:
        return False
    
    sum_of_divisors = sum(i for i in range(1, number // 2 + 1) if number % i == 0)
    return sum_of_divisors == number


def get_divisors(number: int) -> list[int]:
    """Finds all divisors of an integer.

    Args:
        number: The integer to find divisors for.

    Returns:
        A list of divisors.
    """
    if number == 0:
        return []
    num = abs(number)
    return [i for i in range(1, num + 1) if num % i == 0]


def is_power_of_two(number: int) -> bool:
    """Determines if a number is a power of two using bitwise operations.

    Args:
        number: The integer to check.

    Returns:
        True if the number is a power of two, False otherwise.
    """
    if number <= 0:
        return False
    # A power of two in binary is a 1 followed by all 0s (e.g., 8 is 1000).
    # n-1 will be all 1s (e.g., 7 is 0111).
    # n & (n-1) will therefore be 0.
    return (number & (number - 1)) == 0


def calculate_hcf(x: int, y: int) -> int:
    """Calculates the Highest Common Factor (HCF/GCD) of two numbers.

    Args:
        x: The first integer.
        y: The second integer.

    Returns:
        The HCF of x and y.
    """
    return math.gcd(x, y)


def find_pythagorean_triplets(limit: int) -> Generator[tuple[int, int, int], None, None]:
    """Generates Pythagorean triplets (a, b, c) where c is up to a given limit.
    
    Uses Euclid's formula: a = m^2 - n^2, b = 2mn, c = m^2 + n^2.

    Args:
        limit: The maximum value for c.

    Yields:
        A tuple (a, b, c) representing a Pythagorean triplet.
    """
    m = 2
    while True:
        # Smallest c for a given m is with n=1
        if m * m + 1 > limit:
            break
            
        for n in range(1, m):
            a = m * m - n * n
            b = 2 * m * n
            c = m * m + n * n
            
            if c > limit:
                break

            yield tuple(sorted((a, b)) + [c])
        
        m += 1


def is_divisible_by_7_and_multiple_of_5(lower: int, upper: int) -> list[int]:
    """Finds numbers in a range divisible by 7 and a multiple of 5.
    
    Args:
        lower: The start of the range (inclusive).
        upper: The end of the range (inclusive).
    
    Returns:
        A list of numbers satisfying the criteria.
    """
    return [i for i in range(lower, upper + 1) if i % 7 == 0 and i % 5 == 0]


def sum_digits_recursive(number: int) -> int:
    """Calculates the sum of all digits of an integer using recursion.

    Args:
        number: The integer to sum digits of.

    Returns:
        The sum of the digits.
    """
    num_str = str(abs(number))
    if len(num_str) == 0:
        return 0
    return int(num_str[0]) + sum_digits_recursive(int(num_str[1:] or 0))


# --- Bitwise Operations ---

def binary_to_gray(binary_str: str) -> str:
    """Converts a binary string to a Gray codeword string.

    Args:
        binary_str: The input binary string (e.g., "1010").

    Returns:
        The corresponding Gray codeword as a string.
    """
    num = int(binary_str, 2)
    gray_code = num ^ (num >> 1)
    return bin(gray_code)[2:]


def gray_to_binary(gray_str: str) -> str:
    """Converts a Gray codeword string to a binary string.

    Args:
        gray_str: The input Gray codeword string (e.g., "1111").

    Returns:
        The corresponding binary string.
    """
    num = int(gray_str, 2)
    mask = num
    while mask != 0:
        mask >>= 1
        num ^= mask
    return bin(num)[2:]


def clear_rightmost_set_bit(number: int) -> int:
    """Clears the rightmost set (1) bit of a number.

    Args:
        number: The integer to modify.

    Returns:
        The integer with the rightmost set bit cleared.
    """
    return number & (number - 1)


# --- String Operations ---

def replace_char_in_string(text: str, old_char: str, new_char: str) -> str:
    """Replaces all occurrences of a character in a string with another.

    Args:
        text: The original string.
        old_char: The character to be replaced.
        new_char: The character to replace with.

    Returns:
        The modified string.
    """
    return text.replace(old_char, new_char)


def is_substring_present(main_string: str, sub_string: str) -> bool:
    """Checks if a substring is present in a given string.
    
    Args:
        main_string: The string to search within.
        sub_string: The substring to search for.
        
    Returns:
        True if the substring is found, False otherwise.
    """
    return sub_string in main_string


def get_string_length(text: str) -> int:
    """Calculates the length of a string without using len().

    Args:
        text: The input string.

    Returns:
        The length of the string.
    """
    count = 0
    for _ in text:
        count += 1
    return count


def is_anagram(str1: str, str2: str) -> bool:
    """Detects if two strings are anagrams of each other.

    Args:
        str1: The first string.
        str2: The second string.

    Returns:
        True if the strings are anagrams, False otherwise.
    """
    return sorted(str1.lower().replace(" ", "")) == sorted(str2.lower().replace(" ", ""))


def count_word_frequency(text: str) -> dict[str, int]:
    """Counts the frequency of each word in a string.

    Args:
        text: The input string.

    Returns:
        A dictionary with words as keys and their frequencies as values.
    """
    words = text.split()
    frequency = {}
    for word in words:
        frequency[word] = frequency.get(word, 0) + 1
    return frequency


def is_binary_string(text: str) -> bool:
    """Checks if a given string is a binary string.

    Args:
        text: The string to check.

    Returns:
        True if the string contains only '0's and '1's, False otherwise.
    """
    return all(char in '01' for char in text)


def find_string_rotation_count(text: str) -> int:
    """Computes the minimum number of left rotations required to get the same string.
    
    Args:
        text: The input string.
        
    Returns:
        The minimum number of rotations.
    """
    if not text:
        return 0
    
    temp_str = text + text
    n = len(text)
    for i in range(1, n + 1):
        substring = temp_str[i : i + n]
        if text == substring:
            return i
    return n # Should not be reached if text is not empty


# --- List/Collection Operations ---

def get_list_union(list1: list[T], list2: list[T]) -> list[T]:
    """Computes the union of two lists, preserving order from the first list.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        A new list containing unique elements from both lists.
    """
    return list(dict.fromkeys(list1 + list2))


def get_list_intersection(list1: list[T], list2: list[T]) -> list[T]:
    """Finds the intersection of two lists.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        A list containing elements common to both lists.
    """
    set1 = set(list1)
    return [item for item in list2 if item in set1]


def count_occurrences_in_list(data: list[T], item_to_find: T) -> int:
    """Counts the number of times a particular item occurs in a list.

    Args:
        data: The list to search in.
        item_to_find: The item to count.

    Returns:
        The number of occurrences.
    """
    return data.count(item_to_find)


def sort_list_by_element_length(data: list[str]) -> list[str]:
    """Sorts a list of strings according to the length of the elements.

    Args:
        data: The list of strings to sort.

    Returns:
        A new list sorted by string length.
    """
    return sorted(data, key=len)


def find_longest_string_in_list(data: list[str]) -> str | None:
    """Finds the longest string in a list of strings.

    Args:
        data: The list of strings.

    Returns:
        The longest string, or None if the list is empty.
    """
    if not data:
        return None
    return max(data, key=len)


def find_second_largest(numbers: list[int | float]) -> int | float | None:
    """Finds the second largest number in a list.

    Args:
        numbers: A list of numbers.

    Returns:
        The second largest number, or None if the list has fewer than two
        unique elements.
    """
    unique_numbers = sorted(list(set(numbers)))
    if len(unique_numbers) < 2:
        return None
    return unique_numbers[-2]


# --- Dictionary Operations ---

def multiply_dict_values(data: dict[Any, int | float]) -> int | float:
    """Multiplies all the numeric values in a dictionary.

    Args:
        data: A dictionary with numeric values.

    Returns:
        The product of all values.
    """
    if not data:
        return 1
        
    product = 1
    for value in data.values():
        product *= value
    return product


def sum_dict_values(data: dict[Any, int | float]) -> int | float:
    """Sums all the numeric values in a dictionary.

    Args:
        data: A dictionary with numeric values.

    Returns:
        The sum of all values.
    """
    return sum(data.values())


def remove_key_from_dict(data: dict, key_to_remove: Any) -> dict:
    """Removes a given key from a dictionary if it exists.

    Args:
        data: The dictionary to modify.
        key_to_remove: The key to remove.

    Returns:
        The dictionary with the key removed.
    """
    if key_to_remove in data:
        del data[key_to_remove]
    return data

def merge_dictionaries(dict1: dict, dict2: dict) -> dict:
    """Merges two dictionaries. Keys from dict2 will overwrite those in dict1.

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.

    Returns:
        A new merged dictionary.
    """
    return {**dict1, **dict2}


# --- Matrix Operations ---

def add_matrices(matrix_a: Matrix, matrix_b: Matrix) -> Matrix | None:
    """Adds two matrices.

    Args:
        matrix_a: The first matrix.
        matrix_b: The second matrix.

    Returns:
        The resulting sum matrix, or None if dimensions are incompatible.
    """
    if not matrix_a or not matrix_b:
        return None
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])

    if rows_a != rows_b or cols_a != cols_b:
        return None

    result = [[0 for _ in range(cols_a)] for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_a):
            result[i][j] = matrix_a[i][j] + matrix_b[i][j]
    return result


def multiply_matrices(matrix_a: Matrix, matrix_b: Matrix) -> Matrix | None:
    """Multiplies two matrices.

    Args:
        matrix_a: The first matrix (m x n).
        matrix_b: The second matrix (n x p).

    Returns:
        The resulting product matrix (m x p), or None if dimensions are incompatible.
    """
    if not matrix_a or not matrix_b:
        return None
    rows_a, cols_a = len(matrix_a), len(matrix_a[0])
    rows_b, cols_b = len(matrix_b), len(matrix_b[0])

    if cols_a != rows_b:
        return None

    result = [[0 for _ in range(cols_b)] for _ in range(rows_a)]
    for i in range(rows_a):
        for j in range(cols_b):
            for k in range(cols_a): # or rows_b
                result[i][j] += matrix_a[i][k] * matrix_b[k][j]
    return result


def transpose_matrix(matrix: Matrix) -> Matrix | None:
    """Transposes a matrix.

    Args:
        matrix: The matrix to transpose.

    Returns:
        The transposed matrix.
    """
    if not matrix:
        return None
    rows, cols = len(matrix), len(matrix[0])
    
    transposed = [[0 for _ in range(rows)] for _ in range(cols)]
    for i in range(rows):
        for j in range(cols):
            transposed[j][i] = matrix[i][j]
    return transposed


def create_identity_matrix(size: int) -> Matrix:
    """Creates an identity matrix of a given size.

    Args:
        size: The dimension (number of rows/columns) of the matrix.

    Returns:
        An identity matrix.
    """
    return [[1 if i == j else 0 for j in range(size)] for i in range(size)]


# --- Unit Conversions ---

def cm_to_inches(cm: float) -> float:
    """Converts centimeters to inches.
    
    Args:
        cm: The length in centimeters.
        
    Returns:
        The length in inches.
    """
    return cm * 0.393701


def cm_to_feet(cm: float) -> float:
    """Converts centimeters to feet.
    
    Args:
        cm: The length in centimeters.
        
    Returns:
        The length in feet.
    """
    return cm * 0.0328084


# --- File Operations ---

def read_file_lines(filepath: str) -> list[str]:
    """Reads all lines from a file into a list.
    
    Args:
        filepath: The path to the file.
        
    Returns:
        A list of strings, where each string is a line from the file.
    """

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
        return []


def count_lines_in_file(filepath: str) -> int:
    """Counts the number of lines in a text file.
    
    Args:
        filepath: The path to the file.
        
    Returns:
        The total number of lines.
    """
    return len(read_file_lines(filepath))


def count_words_in_file(filepath: str) -> int:
    """Counts the number of words in a text file.
    
    Args:
        filepath: The path to the file.
        
    Returns:
        The total number of words.
    """
    lines = read_file_lines(filepath)
    return sum(len(line.split()) for line in lines)


# --- Classes ---

class Rectangle:
    """A class to represent a rectangle and calculate its area."""
    
    def __init__(self, length: float, breadth: float):
        if length <= 0 or breadth <= 0:
            raise ValueError("Length and breadth must be positive numbers.")
        self.length = length
        self.breadth = breadth

    def area(self) -> float:
        """Calculates the area of the rectangle.
        
        Returns:
            The area of the rectangle.
        """
        return self.length * self.breadth


# ==============================================================================
# MAIN EXECUTION BLOCK
# ==============================================================================

def main():
    """Main function to demonstrate the usage of all refactored functions."""
    
    print("--- Number Theory & Math ---")
    num_perfect = 28
    print(f"Is {num_perfect} a perfect number? {is_perfect(num_perfect)}")
    
    num_divisors = 20
    print(f"Divisors of {num_divisors}: {get_divisors(num_divisors)}")

    num_power_of_two = 16
    print(f"Is {num_power_of_two} a power of two? {is_power_of_two(num_power_of_two)}")

    hcf_x, hcf_y = 54, 24
    print(f"HCF of {hcf_x} and {hcf_y} is: {calculate_hcf(hcf_x, hcf_y)}")

    triplet_limit = 50
    print(f"Pythagorean triplets with c <= {triplet_limit}: {list(find_pythagorean_triplets(triplet_limit))}")

    print(f"Numbers divisible by 7 and multiple of 5 (1-100): {is_divisible_by_7_and_multiple_of_5(1, 100)}")

    sum_digit_num = 12345
    print(f"Sum of digits of {sum_digit_num}: {sum_digits_recursive(sum_digit_num)}")
    
    print("\n--- Bitwise Operations ---")
    binary_val = "1011"
    gray_val = binary_to_gray(binary_val)
    print(f"Binary '{binary_val}' to Gray: '{gray_val}'")
    print(f"Gray '{gray_val}' to Binary: '{gray_to_binary(gray_val)}'")
    
    bit_num = 10 # Binary 1010
    print(f"Clearing rightmost set bit of {bit_num} (1010): {clear_rightmost_set_bit(bit_num)} (binary 1000 is 8)")

    print("\n--- String Operations ---")
    original_str = "hello world, hello python"
    print(f"Replacing 'l' with 'X' in '{original_str}': '{replace_char_in_string(original_str, 'l', 'X')}'")

    print(f"Is 'python' in '{original_str}'? {is_substring_present(original_str, 'python')}")
    
    print(f"Length of '{original_str}' (manual): {get_string_length(original_str)}")
    
    str_a, str_b = "Listen", "Silent"
    print(f"Are '{str_a}' and '{str_b}' anagrams? {is_anagram(str_a, str_b)}")

    print(f"Word frequency of '{original_str}': {count_word_frequency(original_str)}")

    print(f"Is '101010' a binary string? {is_binary_string('101010')}")
    print(f"Is '102010' a binary string? {is_binary_string('102010')}")

    rotation_str = "abcde"
    print(f"Min rotations for '{rotation_str}': {find_string_rotation_count(rotation_str)}")


    print("\n--- List/Collection Operations ---")
    list_1, list_2 = [1, 2, 3, 4], [3, 4, 5, 6]
    print(f"Union of {list_1} and {list_2}: {get_list_union(list_1, list_2)}")
    print(f"Intersection of {list_1} and {list_2}: {get_list_intersection(list_1, list_2)}")

    data_list = [1, 2, 3, 2, 4, 2, 5]
    print(f"Occurrences of 2 in {data_list}: {count_occurrences_in_list(data_list, 2)}")

    str_list = ["apple", "banana", "kiwi", "strawberry", "fig"]
    print(f"List sorted by length: {sort_list_by_element_length(str_list)}")
    print(f"Longest string in list: {find_longest_string_in_list(str_list)}")

    num_list = [10, 50, 20, 90, 85, 90]
    print(f"Second largest in {num_list}: {find_second_largest(num_list)}")


    print("\n--- Dictionary Operations ---")
    my_dict = {'A': 10, 'B': 5, 'C': 2}
    print(f"Product of values in {my_dict}: {multiply_dict_values(my_dict)}")
    print(f"Sum of values in {my_dict}: {sum_dict_values(my_dict)}")
    
    dict1, dict2 = {'a': 1, 'b': 2}, {'b': 3, 'c': 4}
    print(f"Merging {dict1} and {dict2}: {merge_dictionaries(dict1, dict2)}")


    print("\n--- Matrix Operations ---")
    mat_x = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
    mat_y = [[9, 8, 7], [6, 5, 4], [3, 2, 1]]
    print(f"Matrix X + Y: {add_matrices(mat_x, mat_y)}")
    print(f"Matrix X * Y: {multiply_matrices(mat_x, mat_y)}")
    print(f"Transpose of X: {transpose_matrix(mat_x)}")
    print(f"3x3 Identity Matrix: {create_identity_matrix(3)}")

    
    print("\n--- Unit Conversions ---")
    cm_val = 100
    print(f"{cm_val} cm is {cm_to_inches(cm_val):.2f} inches.")
    print(f"{cm_val} cm is {cm_to_feet(cm_val):.2f} feet.")

    
    print("\n--- Class Demonstration ---")
    my_rectangle = Rectangle(length=10, breadth=5)
    print(f"Area of a 10x5 rectangle: {my_rectangle.area()}")

    # Note: File operations and functions requiring specific libraries like
    # pandas or numpy are not demonstrated here to avoid creating files or
    # causing errors if libraries aren't installed. Their usage is clear
    # from their function definitions.

if __name__ == "__main__":
    main()