# -*- coding: utf-8 -*-
"""
This script contains a collection of refactored Python code examples.

Each example from the original script has been encapsulated into a well-documented,
type-hinted, and reusable function following standard Python best practices (PEP 8).

The script is organized into sections:
1.  Mathematical Operations
2.  List/Collection Operations
3.  String Operations
4.  Date & Time Operations
5.  File Operations
6.  Geometric Calculations
7.  Miscellaneous Utilities
8.  A __main__ block to demonstrate the usage of each function.
"""

# 1. Imports - Group all imports at the top of the file.
import datetime
import itertools
import json
import math
import random
import re
import threading
import time
from typing import Any, Dict, List, Tuple, Union, Optional

# ==============================================================================
# 1. Mathematical Operations
# ==============================================================================

def add_numbers(num1: float, num2: float) -> float:
    """Adds two numbers and returns the sum."""
    return num1 + num2

def find_largest(numbers: List[float]) -> float:
    """Finds the largest number in a list of numbers."""
    if not numbers:
        raise ValueError("Input list cannot be empty")
    return max(numbers)

def find_smallest(numbers: List[float]) -> float:
    """Finds the smallest number in a list of numbers."""
    if not numbers:
        raise ValueError("Input list cannot be empty")
    return min(numbers)

def is_prime(number: int) -> bool:
    """
    Checks if a number is a prime number.

    A prime number is a natural number greater than 1 that has no positive
    divisors other than 1 and itself.
    """
    if number < 2:
        return False
    # A number is not prime if it's divisible by any number from 2 to sqrt(number).
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False
    return True

def get_factors(number: int) -> List[int]:
    """Returns a list of all factors for a given positive integer."""
    if number <= 0:
        raise ValueError("Input must be a positive integer")
    return [i for i in range(1, number + 1) if number % i == 0]

def calculate_factorial(n: int) -> int:
    """
    Calculates the factorial of a non-negative integer.

    Note: The `math.factorial()` function from the standard library is
    a more efficient and robust way to do this.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    if n == 0:
        return 1
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def calculate_sum_natural_numbers(n: int) -> int:
    """
    Returns the sum of the first n natural numbers using an efficient formula.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
    # The formula for the sum of the first n natural numbers is n * (n + 1) / 2
    return n * (n + 1) // 2

def calculate_sum_of_squares(n: int) -> int:
    """Calculates and returns the sum of squares of first n natural numbers."""
    if n < 0:
        raise ValueError("Input must be a non-negative integer")
    return sum(i**2 for i in range(1, n + 1))

def calculate_lcm(a: int, b: int) -> int:
    """
    Calculates the Least Common Multiple (LCM) of two integers.
    Uses the formula: LCM(a, b) = |a * b| / GCD(a, b)
    """
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)

def sum_of_digits(number: int) -> int:
    """Returns the sum of the digits of a given non-negative number."""
    if number < 0:
        number = abs(number) # Or raise ValueError if only positive is desired
    
    return sum(int(digit) for digit in str(number))

def is_perfect_number(n: int) -> bool:
    """
    Checks if a number is a perfect number.
    A perfect number is a positive integer that is equal to the sum of its
    proper positive divisors (the sum of its positive divisors excluding the
    number itself).
    """
    if n <= 0:
        return False
    
    # Find sum of proper divisors
    divisor_sum = sum(i for i in range(1, n // 2 + 1) if n % i == 0)
    return divisor_sum == n
    
def calculate_hanoi_steps(disks: int) -> int:
    """
    Calculates the minimum number of steps for the Tower of Hanoi problem.
    The formula is 2^n - 1.
    """
    if disks < 1:
        raise ValueError("Number of disks must be at least 1.")
    return 2**disks - 1

# ==============================================================================
# 2. List/Collection Operations
# ==============================================================================

def merge_lists(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Merges two lists into one."""
    return list1 + list2

def filter_divisible_by(numbers: List[int], divisor: int) -> List[int]:
    """Filters a list to include only numbers divisible by a given divisor."""
    if divisor == 0:
        raise ValueError("Cannot divide by zero.")
    return [num for num in numbers if num % divisor == 0]

def swap_first_last(data: List[Any]) -> List[Any]:
    """Swaps the first and last elements of a list."""
    if len(data) < 2:
        return data  # Nothing to swap
    
    data[0], data[-1] = data[-1], data[0]
    return data

def remove_odd_numbers(numbers: List[int]) -> List[int]:
    """Removes all odd numbers from a list, returning a new list of even numbers."""
    return [num for num in numbers if num % 2 == 0]

def remove_even_numbers(numbers: List[int]) -> List[int]:
    """Removes all even numbers from a list, returning a new list of odd numbers."""
    return [num for num in numbers if num % 2 != 0]

def get_unique_elements(data: List[Any]) -> List[Any]:
    """Returns a list of unique elements, preserving original order."""
    # Using dict.fromkeys preserves order and is efficient for uniqueness.
    return list(dict.fromkeys(data))

def separate_positives_negatives(numbers: List[float]) -> Tuple[List[float], List[float]]:
    """Separates a list of numbers into two lists: positives (and zero) and negatives."""
    positives = [num for num in numbers if num >= 0]
    negatives = [num for num in numbers if num < 0]
    return positives, negatives

def flatten_list(shallow_list: List[List[Any]]) -> List[Any]:
    """Flattens a shallow list of lists into a single list."""
    return list(itertools.chain.from_iterable(shallow_list))

def sort_dictionary_by_key_and_values(d: Dict[Any, List]) -> Dict[Any, List]:
    """
    Sorts a dictionary by its keys, and sorts the list value of each key.
    """
    # Using a dictionary comprehension for a more concise solution.
    return {key: sorted(d[key]) for key in sorted(d.keys())}

# ==============================================================================
# 3. String Operations
# ==============================================================================

def check_number_sign(num: float) -> str:
    """Returns 'Positive', 'Negative', or 'Zero' based on the number's sign."""
    if num > 0:
        return "Positive"
    elif num < 0:
        return "Negative"
    else:
        return "Zero"

def is_palindrome(data: Union[str, int]) -> bool:
    """
    Checks if a given string or integer is a palindrome.
    A palindrome reads the same forwards and backward.
    """
    s = str(data)
    return s == s[::-1]

def replace_vowels(text: str, replacement: str = '_') -> str:
    """Replaces all vowels in a string with a given replacement character."""
    vowels = "AEIOUaeiou"
    for vowel in vowels:
        text = text.replace(vowel, replacement)
    return text

def extract_alphabetic_chars(text: str) -> str:
    """Extracts and returns only the alphabetic characters from a string."""
    return "".join(filter(str.isalpha, text))

def concatenate_strings(*args: str, separator: str = ' ') -> str:
    """Concatenates multiple strings with a specified separator."""
    return separator.join(args)

# ==============================================================================
# 4. Date & Time Operations
# ==============================================================================

def get_current_datetime_str() -> str:
    """Returns the current date and time as a formatted string."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def days_between_dates(d1: datetime.date, d2: datetime.date) -> int:
    """Calculates the number of days between two dates."""
    return abs((d2 - d1).days)

# ==============================================================================
# 5. File Operations
# ==============================================================================

def read_file_contents(filepath: str) -> str:
    """Reads and returns the entire content of a text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: The file '{filepath}' was not found."
    except Exception as e:
        return f"An error occurred: {e}"

# ==============================================================================
# 6. Geometric and Physical Calculations
# ==============================================================================

def calculate_circle_area(radius: float) -> float:
    """Calculates the area of a circle given its radius."""
    if radius < 0:
        raise ValueError("Radius cannot be negative.")
    return math.pi * (radius ** 2)

def calculate_square_area(side: float) -> float:
    """Calculates the area of a square given its side length."""
    return side * side

def calculate_square_perimeter(side: float) -> float:
    """Calculates the perimeter of a square given its side length."""
    return 4 * side

def calculate_rectangle_area(length: float, width: float) -> float:
    """Calculates the area of a rectangle."""
    return length * width

def calculate_rectangle_perimeter(length: float, width: float) -> float:
    """Calculates the perimeter of a rectangle."""
    return 2 * (length + width)

def calculate_cylinder_volume(radius: float, height: float) -> float:
    """Calculates the volume of a cylinder."""
    return calculate_circle_area(radius) * height

def calculate_triangle_area(base: float, height: float) -> float:
    """Calculates the area of a triangle given its base and height."""
    return 0.5 * base * height

def is_valid_triangle(a: float, b: float, c: float, check_by_sides: bool = True) -> bool:
    """
    Checks if a triangle is valid, either by its side lengths or angles.
    - By sides: The sum of any two sides must be greater than the third.
    - By angles: The sum of angles must be exactly 180.
    """
    if a <= 0 or b <= 0 or c <= 0:
        return False # Sides and angles must be positive
        
    if check_by_sides:
        return (a + b > c) and (a + c > b) and (b + c > a)
    else: # Check by angles
        return math.isclose(a + b + c, 180)

def calculate_acceleration(initial_velocity: float, final_velocity: float, time: float) -> float:
    """Calculates acceleration given initial velocity, final velocity, and time."""
    if time == 0:
        raise ValueError("Time cannot be zero for acceleration calculation.")
    return (final_velocity - initial_velocity) / time

# ==============================================================================
# 7. Miscellaneous Utilities
# ==============================================================================

def print_multiplication_table(number: int, terms: int = 10):
    """Prints the multiplication table for a given number."""
    print(f"--- Multiplication Table for {number} ---")
    for i in range(1, terms + 1):
        print(f"{number} x {i} = {number * i}")

def print_powers_of_two(terms: int):
    """Prints the first n powers of 2."""
    if terms < 0:
        return
    print(f"--- First {terms} powers of 2 ---")
    for i in range(terms):
        print(f"2^{i} = {2**i}")

def get_day_of_week(day_number: int) -> Optional[str]:
    """Returns the day of the week for a number from 1 (Sunday) to 7 (Saturday)."""
    days = {
        1: 'Sunday', 2: 'Monday', 3: 'Tuesday', 4: 'Wednesday',
        5: 'Thursday', 6: 'Friday', 7: 'Saturday'
    }
    return days.get(day_number) # .get() safely returns None if key is not found

def calculate_average(*args: float) -> Optional[float]:
    """Calculates the average of a variable number of arguments."""
    if not args:
        return None
    return sum(args) / len(args)

def calculate_compound_interest(principal: float, rate: float, time: float) -> float:
    """Calculates the compound interest."""
    amount = principal * (1 + (rate / 100)) ** time
    return amount - principal

def calculate_simple_interest(principal: float, rate: float, time: float) -> float:
    """Calculates the simple interest."""
    return (principal * rate * time) / 100

def get_bmi_category(height_m: float, weight_kg: float) -> Tuple[float, str]:
    """
    Calculates Body Mass Index (BMI) and returns the value and health category.
    """
    if height_m <= 0 or weight_kg <= 0:
        raise ValueError("Height and weight must be positive values.")
        
    bmi = weight_kg / (height_m ** 2)
    
    if bmi < 18.5:
        category = "Underweight"
    elif 18.5 <= bmi < 25:
        category = "Healthy Weight"
    elif 25 <= bmi < 30:
        category = "Overweight"
    else: # bmi >= 30
        category = "Obese"
        
    return bmi, category

def python_object_to_json_string(obj: Any) -> str:
    """Converts a Python object (e.g., dict) to a formatted JSON string."""
    return json.dumps(obj, sort_keys=True, indent=4)


# ==============================================================================
# Main execution block to demonstrate the functions
# ==============================================================================
if __name__ == "__main__":
    print("--- Running Python Code Examples ---")

    # Mathematical Operations
    print(f"\n1. Sum of 1.5 and 6.3: {add_numbers(1.5, 6.3)}")
    numbers_list = [10, 14, 12, -5]
    print(f"2. Largest in {numbers_list}: {find_largest(numbers_list)}")
    print(f"3. Smallest in {numbers_list}: {find_smallest(numbers_list)}")
    prime_candidate = 337
    print(f"4. Is {prime_candidate} a prime number? {is_prime(prime_candidate)}")
    print(f"5. Factors of 12: {get_factors(12)}")
    print(f"6. Factorial of 13: {calculate_factorial(13)}")
    print(f"7. Sum of first 21 natural numbers: {calculate_sum_natural_numbers(21)}")
    print(f"8. Sum of squares of first 21 natural numbers: {calculate_sum_of_squares(21)}")
    print(f"9. LCM of 12 and 18: {calculate_lcm(12, 18)}")
    print(f"10. Sum of digits of 12345: {sum_of_digits(12345)}")
    print(f"11. Is 28 a perfect number? {is_perfect_number(28)}")
    print(f"12. Min steps for Tower of Hanoi with 3 disks: {calculate_hanoi_steps(3)}")
    print(f"13. Number sign for -55: {check_number_sign(-55)}")

    # List/Collection Operations
    list_a = [1, 2, 3]
    list_b = [4, 5, 6]
    print(f"\n14. Merging {list_a} and {list_b}: {merge_lists(list_a, list_b)}")
    div_list = [11, 45, 74, 132, 21]
    print(f"15. Numbers in {div_list} divisible by 3: {filter_divisible_by(div_list, 3)}")
    swap_list = [1, 2, 3, 4, 5, 6]
    print(f"16. Swapping first/last in {swap_list}: {swap_first_last(swap_list)}")
    num_mix = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    print(f"17. Removing odd numbers from {num_mix}: {remove_odd_numbers(num_mix)}")
    print(f"18. Removing even numbers from {num_mix}: {remove_even_numbers(num_mix)}")
    dup_list = [1, 2, 4, 2, 1, 5, 4, 7, 8]
    print(f"19. Unique elements in {dup_list}: {get_unique_elements(dup_list)}")
    pn_list = [-10, 20, -30, 0, 40]
    pos, neg = separate_positives_negatives(pn_list)
    print(f"20. From {pn_list}, Positives: {pos}, Negatives: {neg}")
    shallow = [[2, 4, 3], [1, 5, 6], [9]]
    print(f"21. Flattening {shallow}: {flatten_list(shallow)}")
    dict_to_sort = {'car': [7, 6, 3], 'bike': [2, 10, 3], 'truck': [19, 4]}
    print(f"22. Sorted dictionary: {sort_dictionary_by_key_and_values(dict_to_sort)}")

    # String Operations
    palindrome_num = 12321
    print(f"\n23. Is {palindrome_num} a palindrome? {is_palindrome(palindrome_num)}")
    vowel_str = "Where is this going? Could you please help me understand!"
    print(f"24. Replacing vowels: {replace_vowels(vowel_str)}")
    alpha_str = "$john.snow#@Got.bad_ending/com"
    print(f"25. Extracting alphabet chars from '{alpha_str}': {extract_alphabetic_chars(alpha_str)}")
    print(f"26. Concatenating strings: {concatenate_strings('Hello', 'world', 'from', 'Python', separator='-')}")

    # Date & Time Operations
    print(f"\n27. Current date and time: {get_current_datetime_str()}")
    date1 = datetime.date(2019, 4, 15)
    date2 = datetime.date(2020, 4, 15)
    print(f"28. Days between {date1} and {date2}: {days_between_dates(date1, date2)}")

    # Geometric and Physical Calculations
    print(f"\n29. Area of a circle with radius 5: {calculate_circle_area(5):.2f}")
    print(f"30. Volume of a cylinder (r=3, h=5): {calculate_cylinder_volume(3, 5):.2f}")
    print(f"31. Area of a triangle (b=11, h=12): {calculate_triangle_area(11, 12)}")
    print(f"32. Is a triangle with sides 3, 4, 5 valid? {is_valid_triangle(3, 4, 5)}")
    print(f"33. Is a triangle with angles 60, 60, 60 valid? {is_valid_triangle(60, 60, 60, check_by_sides=False)}")
    print(f"34. Acceleration (u=0, v=50, t=10): {calculate_acceleration(0, 50, 10)} m/s^2")

    # Miscellaneous Utilities
    print("\n35. Multiplication Table:")
    print_multiplication_table(9, 5) # Print table for 9 up to 5 terms
    print("\n36. Powers of Two:")
    print_powers_of_two(5)
    print(f"37. Day for number 4: {get_day_of_week(4)}")
    print(f"38. Average of 10, 20, 30: {calculate_average(10, 20, 30)}")
    print(f"39. Compound interest (p=1000, r=5, t=2): {calculate_compound_interest(1000, 5, 2):.2f}")
    print(f"40. Simple interest (p=1000, r=5, t=2): {calculate_simple_interest(1000, 5, 2):.2f}")
    bmi_val, bmi_cat = get_bmi_category(height_m=1.75, weight_kg=70)
    print(f"41. BMI for 1.75m, 70kg: {bmi_val:.2f} ({bmi_cat})")
    
    # Lambda examples as simple variables
    multiply = lambda a, b: a * b
    is_even = lambda a: a % 2 == 0
    get_char_from_ascii = lambda code: chr(code)
    print("\n--- Lambda Function Examples ---")
    print(f"42. 5 * 10 = {multiply(5, 10)}")
    print(f"43. Is 4 even? {is_even(4)}")
    print(f"44. Character for ASCII 65: {get_char_from_ascii(65)}")

    # JSON Conversion
    python_dict = {"name": "David", "age": 6, "class": "I"}
    print("\n45. Python Dict to JSON String:")
    print(python_object_to_json_string(python_dict))