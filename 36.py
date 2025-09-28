"""
This file contains a collection of refactored Python code examples.
Each function is documented, type-hinted, and follows PEP 8 standards.
"""
import collections
import itertools
import math
import random
import re
from itertools import groupby
from timeit import Timer
from typing import List, Dict, Any, Tuple, Optional, Union, Generator, Set, Deque


# 56. Find three numbers that sum to zero
def find_three_sum_zero(numbers: List[int]) -> List[List[int]]:
    """Finds all unique triplets in the list which give the sum of zero.

    This solution uses the two-pointer technique after sorting the list
    to achieve an O(n^2) time complexity.

    Args:
        numbers: A list of integers.

    Returns:
        A list of lists, where each inner list is a triplet that sums to zero.
    """
    if len(numbers) < 3:
        return []

    numbers.sort()
    result = []

    for i in range(len(numbers) - 2):
        # Skip duplicate elements to avoid duplicate triplets
        if i > 0 and numbers[i] == numbers[i - 1]:
            continue

        left, right = i + 1, len(numbers) - 1
        while left < right:
            current_sum = numbers[i] + numbers[left] + numbers[right]

            if current_sum == 0:
                result.append([numbers[i], numbers[left], numbers[right]])
                left += 1
                right -= 1
                # Skip duplicates for the left and right pointers
                while left < right and numbers[left] == numbers[left - 1]:
                    left += 1
                while left < right and numbers[right] == numbers[right + 1]:
                    right -= 1
            elif current_sum < 0:
                left += 1
            else:
                right -= 1
    return result


# 57. Find the single number that does not occur twice
def find_single_number_in_duplicates(arr: List[int]) -> int:
    """Finds the single number in a list where every other number appears twice.

    This function uses the XOR bitwise operator. The property of XOR is that
    a ^ a = 0 and a ^ 0 = a. So, all paired numbers cancel out, leaving the
    single number.

    Args:
        arr: A list of integers where one number appears once and others twice.

    Returns:
        The integer that appears only once.
    """
    result = 0
    for num in arr:
        result ^= num
    return result


# 58. Find the single number where others appear three times
def find_single_in_threes(arr: List[int]) -> int:
    """Finds the single element in a list where every other element appears three times.

    This uses a bitwise approach to track bits that have appeared once, twice,
    or three times. `ones` stores bits that appeared once, `twos` stores bits
    that appeared twice. When a bit appears three times, it's reset in both.

    Args:
        arr: A list of integers.

    Returns:
        The integer that appears only once.
    """
    ones, twos = 0, 0
    for x in arr:
        # `ones` holds the bits that have appeared 1, 4, 7, ... times
        # `twos` holds the bits that have appeared 2, 5, 8, ... times
        ones = (ones ^ x) & ~twos
        twos = (twos ^ x) & ~ones
    return ones


# 59. Add digits of a positive integer until the result is a single digit
def add_digits_until_single(num: int) -> int:
    """Repeatedly adds the digits of a positive integer until the result has a single digit.

    This is the "digital root" problem. The formula (num - 1) % 9 + 1 is a
    mathematical shortcut to find the digital root for any positive integer.

    Args:
        num: A positive integer.

    Returns:
        A single-digit integer.
    """
    if num == 0:
        return 0
    return (num - 1) % 9 + 1


# 60. Reverse the digits of an integer
def reverse_integer(x: int) -> int:
    """Reverses the digits of an integer, preserving the sign.

    Args:
        x: An integer.

    Returns:
        The integer with its digits reversed.
    """
    sign = -1 if x < 0 else 1
    abs_x = abs(x)
    
    # Convert to string, reverse, and convert back to integer
    reversed_x = int(str(abs_x)[::-1])
    
    return sign * reversed_x


# 61. Reverse the bits of a 32-bit unsigned integer
def reverse_bits(n: int) -> int:
    """Reverses the bits of a 32-bit unsigned integer.

    Args:
        n: A 32-bit unsigned integer.

    Returns:
        The integer with its bits reversed.
    """
    result = 0
    for i in range(32):
        # Left shift the result to make space for the next bit
        result <<= 1
        # Get the last bit of n and add it to the result
        result |= n & 1
        # Right shift n to process the next bit
        n >>= 1
    return result


# 62. Check if a sequence is an arithmetic progression
def is_arithmetic_progression(num_list: List[Union[int, float]]) -> bool:
    """Checks if a sequence of numbers is an arithmetic progression.

    Args:
        num_list: A list of numbers.

    Returns:
        True if the list is an arithmetic progression, False otherwise.
    """
    if len(num_list) < 2:
        return True
        
    delta = num_list[1] - num_list[0]
    for i in range(2, len(num_list)):
        if num_list[i] - num_list[i - 1] != delta:
            return False
    return True


# 63. Check if a sequence is a geometric progression
def is_geometric_progression(num_list: List[Union[int, float]]) -> bool:
    """Checks if a sequence of numbers is a geometric progression.

    Args:
        num_list: A list of numbers.

    Returns:
        True if the list is a geometric progression, False otherwise.
    """
    if len(num_list) <= 1:
        return True
    
    # Avoid division by zero
    if num_list[0] == 0:
        # If the first element is 0, all other elements must also be 0
        return all(x == 0 for x in num_list)
        
    ratio = num_list[1] / num_list[0]
    for i in range(2, len(num_list)):
        if num_list[i-1] == 0 or num_list[i] / num_list[i-1] != ratio:
            return False
    return True


# 64. Sum of two reversed numbers, with the sum also reversed
def reversed_sum(n1: int, n2: int) -> int:
    """Computes the sum of two reversed numbers and returns the sum in reversed form.

    Args:
        n1: The first integer.
        n2: The second integer.

    Returns:
        The reversed sum of the two reversed numbers.
    """
    rev_n1 = int(str(n1)[::-1])
    rev_n2 = int(str(n2)[::-1])
    total_sum = rev_n1 + rev_n2
    return int(str(total_sum)[::-1])


# 65. Generate the Collatz sequence
def generate_collatz_sequence(n: int) -> List[int]:
    """Generates the Collatz sequence for a given positive integer.

    The sequence is defined as:
    - If n is even, the next number is n / 2.
    - If n is odd, the next number is 3n + 1.
    The sequence stops when it reaches 1.

    Args:
        n: A positive integer.

    Returns:
        A list of integers representing the Collatz sequence.
    """
    if n < 1:
        return []

    sequence = [n]
    while n > 1:
        if n % 2 == 0:
            n //= 2  # Use integer division
        else:
            n = 3 * n + 1
        sequence.append(n)
    return sequence


# 66. Check if two strings are anagrams
def is_anagram(s1: str, s2: str) -> bool:
    """Checks if two strings are anagrams of each other.

    An anagram is a word or phrase formed by rearranging the letters of another.

    Args:
        s1: The first string.
        s2: The second string.

    Returns:
        True if the strings are anagrams, False otherwise.
    """
    # A more efficient way is to use collections.Counter
    return collections.Counter(s1) == collections.Counter(s2)
    # The original way also works but is slightly less efficient:
    # return sorted(s1) == sorted(s2)


# 67. Push all zeros to the end of a list
def move_zeros_to_end(num_list: List[Union[int, float]]) -> List[Union[int, float]]:
    """Moves all zeros in a list to the end, preserving the order of other elements.

    Args:
        num_list: A list of numbers.

    Returns:
        The list with all zeros moved to the end.
    """
    non_zeros = [num for num in num_list if num != 0]
    zeros = [0] * (len(num_list) - len(non_zeros))
    return non_zeros + zeros


# 68. Length of the last word in a string
def length_of_last_word(s: str) -> int:
    """Finds the length of the last word in a string.

    Words are separated by spaces. If there are no words, returns 0.

    Args:
        s: A string.

    Returns:
        The length of the last word.
    """
    words = s.strip().split()
    return len(words[-1]) if words else 0


# 69. Add two binary numbers represented as strings
def add_binary(b1: str, b2: str) -> str:
    """Adds two binary numbers given as strings.

    Args:
        b1: The first binary string.
        b2: The second binary string.

    Returns:
        The sum as a binary string.
    """
    # A much simpler, Pythonic way:
    sum_int = int(b1, 2) + int(b2, 2)
    return bin(sum_int)[2:]
    
    # The original manual implementation is also valid but more verbose.


# 70. Find the number that occurs an odd number of times
def find_odd_occurrence(arr: List[int]) -> int:
    """Finds the number that occurs an odd number of times in a list.
    
    This function assumes there is only one such number and all other numbers
    occur an even number of times. It uses the XOR bitwise operator.

    Args:
        arr: A list of integers.

    Returns:
        The integer that appears an odd number of times.
    """
    result = 0
    for element in arr:
        result ^= element
    return result


# 71. Perform run-length encoding on a string
def run_length_encode(s: str) -> str:
    """Encodes a string using run-length encoding.

    Example: "AAAABBBCCDAAA" -> "4A3B2C1D3A"

    Args:
        s: The input string.

    Returns:
        The encoded string.
    """
    if not s:
        return ""

    encoded_parts = []
    for char, group in itertools.groupby(s):
        count = len(list(group))
        encoded_parts.append(f"{count}{char}")

    return "".join(encoded_parts)


# 72. Product of array elements except self
def product_except_self(nums: List[int]) -> List[int]:
    """Creates an array where each element at index i is the product of all
    numbers in the original array except the one at i.

    This solution avoids division and has a time complexity of O(n).

    Args:
        nums: A list of integers.

    Returns:
        A new list with the calculated products.
    """
    n = len(nums)
    if n == 0:
        return []

    # First pass: calculate prefix products
    prefix_products = [1] * n
    for i in range(1, n):
        prefix_products[i] = prefix_products[i - 1] * nums[i - 1]

    # Second pass: calculate suffix products and multiply with prefix
    suffix_product = 1
    for i in range(n - 1, -1, -1):
        prefix_products[i] *= suffix_product
        suffix_product *= nums[i]

    return prefix_products


# 73. Difference between sum of squares and square of sum
def sum_square_difference(n: int) -> int:
    """Calculates the difference between the square of the sum and the sum of
    the squares of the first n natural numbers.

    Args:
        n: The count of natural numbers (e.g., 1 to n).

    Returns:
        The calculated difference.
    """
    num_range = range(1, n + 1)
    sum_of_nums = sum(num_range)
    square_of_sum = sum_of_nums ** 2
    sum_of_squares = sum(i * i for i in num_range)
    return square_of_sum - sum_of_squares


# 74. Sum of digits of a large number
def sum_of_digits_power(base: int, exponent: int) -> int:
    """Computes the sum of the digits of base raised to the power of exponent.

    Args:
        base: The base number.
        exponent: The exponent.

    Returns:
        The sum of the digits of the result.
    """
    number = base ** exponent
    return sum(int(digit) for digit in str(number))


# 75. Sum of multiples of 3 or 5 below a limit
def sum_multiples_of_3_or_5(limit: int) -> int:
    """Computes the sum of all multiples of 3 or 5 below a given limit.

    Args:
        limit: The upper bound (exclusive).

    Returns:
        The sum of the multiples.
    """
    return sum(i for i in range(1, limit) if i % 3 == 0 or i % 5 == 0)


# 76. Convert an integer to a string in any base
def int_to_base_string(n: int, base: int) -> str:
    """Converts a non-negative integer to its string representation in a given base.

    Args:
        n: The integer to convert.
        base: The target base (2-16).

    Returns:
        The string representation in the given base.
    """
    if not 2 <= base <= 16:
        raise ValueError("Base must be between 2 and 16")
    if n == 0:
        return "0"

    conversion_chars = "0123456789ABCDEF"
    if n < base:
        return conversion_chars[n]
    else:
        return int_to_base_string(n // base, base) + conversion_chars[n % base]


# 77. Calculate the geometric sum
def geometric_sum(n: int) -> float:
    """Calculates the geometric sum: 1 + 1/2 + 1/4 + ... + 1/(2^n).

    Args:
        n: A non-negative integer.

    Returns:
        The geometric sum.
    """
    if n < 0:
        return 0
    else:
        return 1 / (2 ** n) + geometric_sum(n - 1)


# 78. Find the greatest common divisor (GCD) using recursion
def gcd_recursive(a: int, b: int) -> int:
    """Finds the greatest common divisor (GCD) of two integers using the Euclidean algorithm.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The GCD of a and b.
    """
    if b == 0:
        return a
    else:
        return gcd_recursive(b, a % b)


# 79. Find numbers divisible by 7 but not by 5 in a range
def find_divisible_by_7_not_5(start: int, end: int) -> List[int]:
    """Finds all numbers in a given range (inclusive) that are divisible by 7
    but not a multiple of 5.

    Args:
        start: The start of the range.
        end: The end of the range.

    Returns:
        A list of matching integers.
    """
    return [i for i in range(start, end + 1) if i % 7 == 0 and i % 5 != 0]


# 80. Find the roots of a quadratic equation
def solve_quadratic(a: float, b: float, c: float) -> Optional[Tuple[Union[float, complex], ...]]:
    """Calculates the roots of a quadratic equation ax^2 + bx + c = 0.

    Args:
        a: The first coefficient.
        b: The second coefficient.
        c: The third coefficient.

    Returns:
        A tuple containing the roots. Returns None if 'a' is 0 (not a quadratic equation).
    """
    if a == 0:
        return None  # Not a quadratic equation

    discriminant = (b**2) - (4 * a * c)

    if discriminant >= 0:
        # Real roots
        sqrt_d = math.sqrt(discriminant)
        r1 = (-b + sqrt_d) / (2 * a)
        r2 = (-b - sqrt_d) / (2 * a)
        return r1, r2
    else:
        # Imaginary roots
        real_part = -b / (2 * a)
        imag_part = math.sqrt(-discriminant) / (2 * a)
        return complex(real_part, imag_part), complex(real_part, -imag_part)


# 81. Convert a bytearray to a hexadecimal string
def bytearray_to_hex(byte_array: bytearray) -> str:
    """Converts a bytearray to its hexadecimal string representation.

    Args:
        byte_array: The bytearray to convert.

    Returns:
        The hexadecimal string.
    """
    return ''.join(f'{x:02x}' for x in byte_array)


# 82. Count substrings with the same first and last characters
def count_substrings_with_equal_ends(s: str) -> int:
    """Counts the number of substrings that start and end with the same character.
    
    A more efficient approach than O(n^2) is to count character frequencies.
    For a character appearing 'k' times, it can form k*(k+1)/2 such substrings.

    Args:
        s: The input string.

    Returns:
        The total count of such substrings.
    """
    total = 0
    char_counts = collections.Counter(s)
    for char in char_counts:
        k = char_counts[char]
        total += k * (k + 1) // 2
    return total


# 83. Move all spaces to the front of a string
def move_spaces_to_front(s: str) -> str:
    """Moves all space characters to the front of a string.

    Args:
        s: The input string.

    Returns:
        A new string with spaces moved to the front.
    """
    no_spaces = [char for char in s if char != ' ']
    space_count = len(s) - len(no_spaces)
    return ' ' * space_count + ''.join(no_spaces)


# 84. Find the maximum length of consecutive 0s in a binary string
def max_consecutive_zeros(binary_str: str) -> int:
    """Finds the maximum length of a consecutive run of '0's in a binary string.

    Args:
        binary_str: A string containing '0's and '1's.

    Returns:
        The maximum length of consecutive zeros.
    """
    if '0' not in binary_str:
        return 0
    return max(len(s) for s in binary_str.split('1'))


# 85. Iterate over elements repeating each as many times as its count
def expand_counter_elements(data: Dict[Any, int]) -> List[Any]:
    """Given a dictionary of item counts, creates a list where each item is
    repeated by its count.

    Args:
        data: A dictionary mapping items to their counts.

    Returns:
        A list of expanded elements.
    """
    c = collections.Counter(data)
    return list(c.elements())


# 86. Find the second smallest number in a list
def find_second_smallest(numbers: List[Union[int, float]]) -> Optional[Union[int, float]]:
    """Finds the second smallest number in a list of numbers.

    Args:
        numbers: A list of numbers.

    Returns:
        The second smallest number, or None if there isn't one.
    """
    unique_numbers = sorted(list(set(numbers)))
    if len(unique_numbers) < 2:
        return None
    return unique_numbers[1]


# 87. Check if a list contains a sublist
def contains_sublist(main_list: List[Any], sub_list: List[Any]) -> bool:
    """Checks if a list contains another list as a contiguous sublist.

    Args:
        main_list: The list to search within.
        sub_list: The sublist to search for.

    Returns:
        True if the sublist is found, False otherwise.
    """
    if not sub_list:
        return True
    if not main_list or len(sub_list) > len(main_list):
        return False
        
    # Convert lists to a string-like representation to use string searching
    # This is a simple but effective approach for hashable elements
    str_main = ",".join(map(str, main_list))
    str_sub = ",".join(map(str, sub_list))
    
    return str_sub in str_main


# 88. Generate groups of five consecutive numbers in a list
def generate_consecutive_groups(num_groups: int, group_size: int) -> List[List[int]]:
    """Generates a list of lists, with groups of consecutive numbers.

    Args:
        num_groups: The number of groups to generate.
        group_size: The size of each group.

    Returns:
        A list of lists with consecutive numbers.
    """
    return [[group_size * i + j for j in range(1, group_size + 1)] for i in range(num_groups)]


# 89. Find the list with the highest sum of elements in a list of lists
def find_list_with_max_sum(lists: List[List[Union[int, float]]]) -> Optional[List[Union[int, float]]]:
    """Finds the sublist with the maximum sum of its elements.

    Args:
        lists: A list of lists of numbers.

    Returns:
        The sublist with the highest sum, or None if the input is empty.
    """
    if not lists:
        return None
    return max(lists, key=sum)


# 90. Calculate the depth of a dictionary
def get_dict_depth(d: Dict[Any, Any]) -> int:
    """Recursively calculates the maximum depth of a nested dictionary.

    Args:
        d: The dictionary to check.

    Returns:
        An integer representing the depth.
    """
    if not isinstance(d, dict) or not d:
        return 0
    return 1 + max(get_dict_depth(v) for v in d.values())


# 91. Pack consecutive duplicates into sublists
def pack_consecutive_duplicates(items: List[Any]) -> List[List[Any]]:
    """Packs consecutive duplicates of list elements into sublists.

    Example: [0, 0, 1, 2, 2, 2] -> [[0, 0], [1], [2, 2, 2]]

    Args:
        items: A list of items.

    Returns:
        A list of lists with grouped consecutive duplicates.
    """
    return [list(group) for key, group in groupby(items)]


# 92. Modified run-length encoding for a list
def modified_run_length_encode(items: List[Any]) -> List[Any]:
    """Creates a modified run-length encoding from a list.
    Single elements are kept as is, while runs of duplicates are
    represented as [count, element].

    Example: [1, 1, 2, 3, 3, 3] -> [[2, 1], 2, [3, 3]]

    Args:
        items: A list of items.

    Returns:
        The modified run-length encoded list.
    """
    def format_group(group: List[Any]) -> Any:
        if len(group) > 1:
            return [len(group), group[0]]
        else:
            return group[0]

    return [format_group(list(group)) for key, group in groupby(items)]


# 93. Create a multidimensional list of zeros
def create_zero_matrix(rows: int, cols: int) -> List[List[int]]:
    """Creates a 2D list (matrix) of a given size, filled with zeros.

    Args:
        rows: The number of rows.
        cols: The number of columns.

    Returns:
        A list of lists representing the zero matrix.
    """
    return [[0 for _ in range(cols)] for _ in range(rows)]


# 94. Check if a nested list is a subset of another
def is_nested_subset(list1: List[List[Any]], list2: List[List[Any]]) -> bool:
    """Checks if every list in list2 is also present in list1.

    Args:
        list1: The potential superset list.
        list2: The potential subset list.

    Returns:
        True if list2 is a subset of list1, False otherwise.
    """
    # For this to work reliably, the order within sublists matters.
    # To handle unordered sublists, we would need to convert them to a canonical form.
    set1 = {tuple(sorted(sublist)) for sublist in list1}
    set2 = {tuple(sorted(sublist)) for sublist in list2}
    return set2.issubset(set1)


# 95. Find all permutations with repetition
def permutations_with_repetition(s: str, length: int) -> List[Tuple[str, ...]]:
    """Finds all permutations of a given length from the characters of a string,
    with repetition allowed.

    Args:
        s: The string of characters to use.
        length: The length of each permutation.

    Returns:
        A list of tuples, where each tuple is a permutation.
    """
    chars = list(s)
    return list(itertools.product(chars, repeat=length))


# 96. Find the starting index of a substring
def find_substring_index(main_str: str, sub_str: str) -> Union[int, str]:
    """Finds the starting index of a substring within a string.

    Args:
        main_str: The string to search in.
        sub_str: The substring to search for.

    Returns:
        The starting index if found, otherwise the string 'Not found'.
    """
    # The built-in `find` method is the standard way to do this.
    index = main_str.find(sub_str)
    return index if index != -1 else 'Not found'


# 97. Find the smallest multiple of the first n numbers
def smallest_multiple(n: int) -> int:
    """Finds the smallest positive number that is evenly divisible by all
    of the numbers from 1 to n (Least Common Multiple).

    Args:
        n: The upper limit of the range of divisors.

    Returns:
        The smallest multiple.
    """
    if n <= 0:
        return 0
    lcm = 1
    for i in range(1, n + 1):
        lcm = (lcm * i) // gcd_recursive(lcm, i)
    return lcm


# 98. Multiply two integers without using the '*' operator
def multiply_recursive(x: int, y: int) -> int:
    """Multiplies two integers using recursion and addition, without the '*' operator.

    Args:
        x: The first integer.
        y: The second integer.

    Returns:
        The product of x and y.
    """
    if y == 0:
        return 0
    if y > 0:
        return x + multiply_recursive(x, y - 1)
    if y < 0:
        return -multiply_recursive(x, -y)
    return 0 # Should be unreachable, but linters appreciate it


# 99. Calculate distance between two points using latitude and longitude
def calculate_haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate the distance between two points on Earth using their latitudes and longitudes.

    Args:
        lat1: Latitude of the first point in degrees.
        lon1: Longitude of the first point in degrees.
        lat2: Latitude of the second point in degrees.
        lon2: Longitude of the second point in degrees.

    Returns:
        The distance in kilometers.
    """
    # Earth radius in kilometers
    earth_radius_km = 6371.01

    # Convert degrees to radians
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)

    # Haversine formula
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad

    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = earth_radius_km * c
    return distance

# 100. Roman to Integer and Integer to Roman conversion classes
class RomanConverter:
    """A class to handle conversions between Roman numerals and integers."""

    def __init__(self):
        """Initializes the RomanConverter."""
        self.rom_to_int_map = {'I': 1, 'V': 5, 'X': 10, 'L': 50, 'C': 100, 'D': 500, 'M': 1000}
        self.int_to_rom_map = [
            (1000, "M"), (900, "CM"), (500, "D"), (400, "CD"),
            (100, "C"), (90, "XC"), (50, "L"), (40, "XL"),
            (10, "X"), (9, "IX"), (5, "V"), (4, "IV"),
            (1, "I")
        ]

    def roman_to_int(self, s: str) -> int:
        """Converts a Roman numeral string to an integer.

        Args:
            s: The Roman numeral string.

        Returns:
            The integer equivalent.
        """
        int_val = 0
        for i in range(len(s)):
            current_val = self.rom_to_int_map[s[i]]
            # If the next numeral is larger, it's a subtractive case (e.g., IV, IX)
            if i + 1 < len(s) and self.rom_to_int_map[s[i + 1]] > current_val:
                int_val -= current_val
            else:
                int_val += current_val
        return int_val

    def int_to_roman(self, num: int) -> str:
        """Converts an integer to a Roman numeral string.

        Args:
            num: An integer (must be between 1 and 3999).

        Returns:
            The Roman numeral string.
        """
        if not 0 < num < 4000:
            raise ValueError("Input must be between 1 and 3999")
        
        roman_num = ''
        for val, syb in self.int_to_rom_map:
            while num >= val:
                roman_num += syb
                num -= val
        return roman_num

# 101. Merge two sorted lists
def merge_sorted_lists(list1: List[int], list2: List[int]) -> List[int]:
    """Merges two sorted lists into a single sorted list.

    Args:
        list1: The first sorted list.
        list2: The second sorted list.

    Returns:
        A new merged and sorted list.
    """
    merged = []
    i, j = 0, 0

    while i < len(list1) and j < len(list2):
        if list1[i] < list2[j]:
            merged.append(list1[i])
            i += 1
        else:
            merged.append(list2[j])
            j += 1

    # Append the remaining elements from whichever list is not exhausted
    merged.extend(list1[i:])
    merged.extend(list2[j:])
    
    return merged

# 102. Right rotate a list by n positions
def right_rotate(lst: List[Any], n: int) -> List[Any]:
    """Rotates a list to the right by n positions.

    Args:
        lst: The list to rotate.
        n: The number of positions to rotate.

    Returns:
        The rotated list.
    """
    if not lst:
        return []
    n = n % len(lst)
    return lst[-n:] + lst[:-n]

# 103. Rearrange a list with negative elements first, then positive
def rearrange_negatives_positives(lst: List[Union[int, float]]) -> List[Union[int, float]]:
    """Rearranges list elements so that all negative numbers appear on the left
    and all positive numbers (and zero) appear on the right.

    Args:
        lst: The list of numbers to rearrange.

    Returns:
        The rearranged list.
    """
    # A simple partition using list comprehensions is very readable
    negatives = [x for x in lst if x < 0]
    positives_and_zeros = [x for x in lst if x >= 0]
    return negatives + positives_and_zeros

# 104. Binary Search in a sorted list
def binary_search(sorted_list: List[Any], element: Any) -> int:
    """Performs a binary search for an element in a sorted list.

    Args:
        sorted_list: A list of elements sorted in ascending order.
        element: The element to search for.

    Returns:
        The index of the element if found, otherwise -1.
    """
    low, high = 0, len(sorted_list) - 1
    
    while low <= high:
        mid = (low + high) // 2
        if sorted_list[mid] == element:
            return mid
        elif sorted_list[mid] < element:
            low = mid + 1
        else:
            high = mid - 1
            
    return -1

# 105. Custom Exception Class
class CustomError(Exception):
    """A custom exception class that takes a message."""
    def __init__(self, message: str):
        self.message = message
        super().__init__(self.message)

# Main execution block to demonstrate the functions
if __name__ == "__main__":
    print("--- Demonstrating Refactored Functions ---\n")

    # 56. find_three_sum_zero
    print(f"56. Triplets that sum to zero in [-1, 0, 1, 2, -1, -4]: {find_three_sum_zero([-1, 0, 1, 2, -1, -4])}")

    # 59. add_digits_until_single
    print(f"59. Single digit sum of 38: {add_digits_until_single(38)}")

    # 60. reverse_integer
    print(f"60. Reverse of -123: {reverse_integer(-123)}")

    # 65. generate_collatz_sequence
    print(f"65. Collatz sequence for 6: {generate_collatz_sequence(6)}")

    # 66. is_anagram
    print(f"66. Is 'listen' an anagram of 'silent'? {is_anagram('listen', 'silent')}")

    # 71. run_length_encode
    print(f"71. RLE of 'AAAABBBCCDAAA': {run_length_encode('AAAABBBCCDAAA')}")
    
    # 72. product_except_self
    print(f"72. Product except self for [1, 2, 3, 4]: {product_except_self([1, 2, 3, 4])}")
    
    # 73. sum_square_difference
    print(f"73. Sum square difference for first 100 numbers: {sum_square_difference(100)}")
    
    # 75. sum_multiples_of_3_or_5
    print(f"75. Sum of multiples of 3 or 5 below 1000: {sum_multiples_of_3_or_5(1000)}")
    
    # 79. find_divisible_by_7_not_5
    numbers_found = find_divisible_by_7_not_5(2000, 2100)
    print(f"79. Numbers divisible by 7 but not 5 (2000-2100): {numbers_found[:5]}...")

    # 80. solve_quadratic
    print(f"80. Roots of x^2 - 5x + 6 = 0: {solve_quadratic(1, -5, 6)}")
    print(f"80. Roots of x^2 + 4x + 5 = 0: {solve_quadratic(1, 4, 5)}")
    
    # 99. calculate_haversine_distance (Paris to New York)
    paris_lat, paris_lon = 48.8566, 2.3522
    ny_lat, ny_lon = 40.7128, -74.0060
    dist = calculate_haversine_distance(paris_lat, paris_lon, ny_lat, ny_lon)
    print(f"99. Distance between Paris and New York: {dist:.2f} km")

    # 100. RomanConverter
    converter = RomanConverter()
    print(f"100. Roman 'MCMXCIV' to Int: {converter.roman_to_int('MCMXCIV')}")
    print(f"100. Int 1994 to Roman: {converter.int_to_roman(1994)}")
    
    # 101. merge_sorted_lists
    l1, l2 = [1, 3, 5], [2, 4, 6]
    print(f"101. Merging {l1} and {l2}: {merge_sorted_lists(l1, l2)}")
    
    # 104. binary_search
    sorted_data = [2, 5, 7, 8, 11, 12]
    print(f"104. Searching for 11 in {sorted_data}: Index {binary_search(sorted_data, 11)}")
    
    # 105. Custom Exception
    try:
        raise CustomError("This is a test of the custom exception.")
    except CustomError as e:
        print(f"105. Caught a custom exception: {e.message}")