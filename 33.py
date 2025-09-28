# 1_math_operations.py
"""A collection of functions for various mathematical operations."""

import math
import random
import sys
from typing import List, Union

# --- Number Properties ---

def is_power_of_two(n: int) -> bool:
    """
    Checks if a number is a power of two using a bitwise operation.

    Args:
        n: An integer.

    Returns:
        True if n is a power of two, False otherwise.
    """
    if n <= 0:
        return False
    # A power of two in binary is `1` followed by zeros (e.g., 8 is 1000).
    # `n-1` will be all ones (e.g., 7 is 0111).
    # The bitwise AND of these two numbers will always be 0.
    return (n & (n - 1)) == 0

def is_even(num: int) -> bool:
    """Checks if a number is even."""
    return num % 2 == 0

def is_odd(num: int) -> bool:
    """Checks if a number is odd."""
    return num % 2 != 0

def is_leap_year(year: int) -> bool:
    """
    Checks if a given year is a leap year.

    A year is a leap year if it is divisible by 4, except for end-of-century
    years, which must be divisible by 400.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def get_prime_numbers(start: int, end: int) -> List[int]:
    """
    Finds all prime numbers within a given range.

    Args:
        start: The starting number of the range.
        end: The ending number of the range.

    Returns:
        A list of prime numbers found in the range.
    """
    primes = []
    for num in range(start, end + 1):
        if num > 1:
            # Check for factors from 2 up to the square root of num
            for i in range(2, int(math.sqrt(num)) + 1):
                if (num % i) == 0:
                    break  # Not a prime number
            else:
                primes.append(num)
    return primes

def get_factors(num: int) -> List[int]:
    """Finds all factors of a given number."""
    if num <= 0:
        return []
    return [i for i in range(1, num + 1) if num % i == 0]

# --- Basic Arithmetic ---

def add(x: float, y: float) -> float:
    """Adds two numbers."""
    return x + y

def subtract(x: float, y: float) -> float:
    """Subtracts the second number from the first."""
    return x - y

def multiply(x: float, y: float) -> float:
    """Multiplies two numbers."""
    return x * y

def divide(x: float, y: float) -> Union[float, str]:
    """Divides the first number by the second."""
    if y == 0:
        return "Error: Cannot divide by zero."
    return x / y

def factorial(num: int) -> int:
    """
    Calculates the factorial of a non-negative integer.
    """
    if num < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    return math.factorial(num)

# --- Geometry ---

def circle_circumference(radius: float) -> float:
    """Calculates the circumference of a circle."""
    return 2 * math.pi * radius

def rectangle_area(length: float, width: float) -> float:
    """Calculates the area of a rectangle."""
    return length * width

def square_area(side: float) -> float:
    """Calculates the area of a square."""
    return side * side

# --- Advanced Operations ---

def compute_hcf(x: int, y: int) -> int:
    """
    Computes the Highest Common Factor (HCF/GCD) of two numbers
    using the Euclidean algorithm.
    """
    while y:
        x, y = y, x % y
    return x

def compute_lcm(x: int, y: int) -> int:
    """
    Computes the Lowest Common Multiple (LCM) of two numbers.
    Formula: LCM(a, b) = |a * b| / HCF(a, b)
    """
    if x == 0 or y == 0:
        return 0
    return abs(x * y) // compute_hcf(x, y)

# --- Main execution block for demonstration ---
if __name__ == "__main__":
    print("--- Power of Two Check ---")
    num_to_check = 16
    if is_power_of_two(num_to_check):
        print(f"{num_to_check} is a power of two.")
    else:
        print(f"{num_to_check} is not a power of two.")
    print("-" * 20)

    print("--- Prime Numbers in Range ---")
    primes = get_prime_numbers(10, 50)
    print(f"Primes between 10 and 50: {primes}")
    print("-" * 20)
    
    print("--- Geometry Calculations ---")
    print(f"Circumference of a circle with radius 10: {circle_circumference(10):.2f}")
    print(f"Area of a 10x5 rectangle: {rectangle_area(10, 5)}")
    print("-" * 20)

    print("--- HCF and LCM ---")
    num1, num2 = 54, 24
    hcf = compute_hcf(num1, num2)
    lcm = compute_lcm(num1, num2)
    print(f"HCF of {num1} and {num2} is: {hcf}")
    print(f"LCM of {num1} and {num2} is: {lcm}")
    print("-" * 20)

    print("--- Factorial ---")
    print(f"Factorial of 5 is: {factorial(5)}")
    print("-" * 20)







# 2_string_manipulation.py
"""A collection of functions for string manipulation and processing."""

import string
from typing import List, Set, Dict

def reverse_words(sentence: str) -> str:
    """
    Reverses the order of words in a sentence.

    Args:
        sentence: The input string.

    Returns:
        A string with the words in reverse order.
    """
    words = sentence.split()
    return " ".join(reversed(words))

def filter_words_by_length(sentence: str, min_length: int = 0, max_length: int = float('inf')) -> List[str]:
    """
    Filters words in a sentence based on their length.

    Args:
        sentence: The input string.
        min_length: The minimum length of words to keep.
        max_length: The maximum length of words to keep.

    Returns:
        A list of words that meet the length criteria.
    """
    words = sentence.split()
    return [word for word in words if min_length <= len(word) <= max_length]

def get_punctuations_used(text: str) -> Set[str]:
    """
    Returns a set of all punctuation characters used in a string.
    """
    return {char for char in text if char in string.punctuation}

def remove_punctuations(text: str) -> str:
    """
    Removes all punctuation characters from a string.
    """
    return text.translate(str.maketrans('', '', string.punctuation))

def replace_words_with_length(sentence: str) -> str:
    """
    Replaces each word in a sentence with its length.
    """
    words = sentence.split()
    lengths = [str(len(word)) for word in words]
    return " ".join(lengths)

def is_palindrome(text: str) -> bool:
    """
    Checks if a string is a palindrome (reads the same forwards and backwards).
    
    Note: This is a case-sensitive check. For a case-insensitive check,
    pre-process the string with .lower().
    """
    return text == text[::-1]

def find_least_frequent_char(text: str) -> str:
    """
    Finds the least frequent character in a string.
    
    Returns the first one found in case of a tie. Ignores spaces.
    """
    text = text.replace(" ", "")
    if not text:
        return ""
    
    freq = {}
    for char in text:
        freq[char] = freq.get(char, 0) + 1
        
    return min(freq, key=freq.get)

def count_vowels(text: str) -> Dict[str, int]:
    """
    Counts the occurrences of each vowel in a string (case-insensitive).
    """
    vowels = "aeiou"
    text_lower = text.lower()
    vowel_counts = {vowel: 0 for vowel in vowels}
    
    for char in text_lower:
        if char in vowel_counts:
            vowel_counts[char] += 1
            
    return vowel_counts

# --- Main execution block for demonstration ---
if __name__ == "__main__":
    sentence = "The quick, brown fox jumps over the lazy dog!"
    
    print(f"Original sentence: '{sentence}'")
    print("-" * 20)
    
    print(f"Reversed words: '{reverse_words(sentence)}'")
    
    print(f"Words with min length 4: {filter_words_by_length(sentence, min_length=4)}")
    
    print(f"Words with max length 3: {filter_words_by_length(sentence, max_length=3)}")

    print(f"Punctuations used: {get_punctuations_used(sentence)}")
    
    print(f"Sentence with lengths: '{replace_words_with_length(sentence)}'")

    palindrome_test = "madam"
    print(f"Is '{palindrome_test}' a palindrome? {is_palindrome(palindrome_test)}")
    
    print(f"Least frequent character in 'programming is fun': '{find_least_frequent_char('programming is fun')}'")
    
    print(f"Vowel counts in '{sentence}': {count_vowels(sentence)}")







# 3_list_operations.py
"""A collection of functions for list manipulation and processing."""

import random
from typing import List, Any, TypeVar

T = TypeVar('T') # Generic type variable for functions that work on any type

def get_list_length_recursive(items: List[Any]) -> int:
    """
    Calculates the length of a list using recursion.

    Note: This is for demonstration of recursion. In practice,
    always use the built-in `len()` function.
    """
    if not items:
        return 0
    return 1 + get_list_length_recursive(items[1:])

def shuffle_list(items: List[T]) -> List[T]:
    """
    Shuffles a list in place and returns it.

    Args:
        items: The list to be shuffled.

    Returns:
        The same list, now shuffled.
    """
    random.shuffle(items)
    return items

def make_all_positive(numbers: List[float]) -> List[float]:
    """Converts all numbers in a list to their absolute values."""
    return [abs(num) for num in numbers]

def make_all_negative(numbers: List[float]) -> List[float]:
    """Converts all numbers in a list to their negative absolute values."""
    return [-abs(num) for num in numbers]

def clone_list(items: List[T]) -> List[T]:
    """Creates a shallow copy of a list."""
    return items[:]

def reverse_list(items: List[T]) -> List[T]:
    """Returns a new list with the items in reverse order."""
    return items[::-1]

def find_common_items(list1: List[T], list2: List[T]) -> List[T]:
    """
    Finds common items between two lists.
    
    Returns:
        A list of items present in both lists. The order is not guaranteed.
    """
    return list(set(list1) & set(list2))

def contains_sublist(main_list: List[T], sub_list: List[T]) -> bool:
    """
    Checks if a list contains another list as a contiguous sublist.
    """
    if not sub_list:
        return True
    if not main_list:
        return False
        
    # Convert lists to strings for simple substring search
    # This is a clever trick for hashable items, but may be slow for large lists.
    str_main = ",".join(map(str, main_list))
    str_sub = ",".join(map(str, sub_list))
    
    return str_sub in str_main

# --- Main execution block for demonstration ---
if __name__ == "__main__":
    my_numbers = [1, -2, 3, 0, -4, 5]
    my_items = ['apple', 'banana', 'cherry']
    
    print(f"Original numbers: {my_numbers}")
    print(f"Length (recursive): {get_list_length_recursive(my_numbers)}")
    print(f"All positive: {make_all_positive(my_numbers)}")
    print(f"All negative: {make_all_negative(my_numbers)}")
    print("-" * 20)
    
    print(f"Original items: {my_items}")
    shuffled = shuffle_list(my_items[:]) # Shuffle a copy
    print(f"Shuffled items: {shuffled}")
    cloned = clone_list(my_items)
    print(f"Cloned items: {cloned}")
    reversed_items = reverse_list(my_items)
    print(f"Reversed items: {reversed_items}")
    print("-" * 20)
    
    list1 = [1, 2, 3, 4, 5]
    list2 = [4, 5, 6, 7, 8]
    print(f"Common items between {list1} and {list2}: {find_common_items(list1, list2)}")
    print("-" * 20)
    
    main_list = [1, 2, 3, 4, 5, 6]
    sub1 = [3, 4, 5]
    sub2 = [1, 3, 5]
    print(f"Does {main_list} contain {sub1}? {contains_sublist(main_list, sub1)}")
    print(f"Does {main_list} contain {sub2}? {contains_sublist(main_list, sub2)}")


