"""
dictionaries_examples.py

A collection of examples demonstrating the creation, manipulation, and advanced
usage of dictionaries in Python.
"""
from typing import Any, Dict, List, Tuple


def demonstrate_dictionary_creation():
    """Shows various ways to create a dictionary."""
    print("--- Demonstrating Dictionary Creation ---")

    # 1. With integer keys
    int_key_dict = {1: 'Geeks', 2: 'For', 3: 'Geeks'}
    print(f"\nDictionary with integer keys: {int_key_dict}")

    # 2. With mixed keys
    mixed_key_dict = {'Name': 'Geeks', 1: [1, 2, 3, 4]}
    print(f"\nDictionary with mixed keys: {mixed_key_dict}")

    # 3. An empty dictionary
    empty_dict = {}
    print(f"\nEmpty dictionary: {empty_dict}")

    # 4. Using the dict() constructor with a dictionary
    dict_from_constructor = dict({1: 'Geeks', 2: 'For', 3: 'Geeks'})
    print(f"\nDictionary from dict() constructor: {dict_from_constructor}")

    # 5. Using the dict() constructor with a list of pairs (tuples)
    dict_from_pairs = dict([(1, 'Geeks'), (2, 'For')])
    print(f"\nDictionary from item pairs: {dict_from_pairs}")

    # 6. A nested dictionary
    nested_dict = {
        1: 'Geeks',
        2: 'For',
        3: {'A': 'Welcome', 'B': 'To', 'C': 'Geeks'}
    }
    print(f"\nNested dictionary: {nested_dict}\n")


def demonstrate_dictionary_access():
    """Shows how to access elements in a dictionary."""
    print("--- Demonstrating Dictionary Access ---")

    # 1. Accessing with .get() (safe access)
    simple_dict = {1: 'Geeks', 'name': 'For', 3: 'Geeks'}
    value = simple_dict.get(3)
    print(f"\nAccessing key 3 using .get(): {value}")
    # .get() returns None if the key doesn't exist, preventing an error
    value_none = simple_dict.get(99)
    print(f"Accessing non-existent key 99 with .get(): {value_none}")

    # 2. Accessing nested dictionary elements
    nested_dict = {
        'Dict1': {1: 'Geeks'},
        'Dict2': {'Name': 'For'}
    }
    print(f"\nAccessing Dict1: {nested_dict['Dict1']}")
    print(f"Accessing element [1] from Dict1: {nested_dict['Dict1'][1]}")
    print(f"Accessing element ['Name'] from Dict2: {nested_dict['Dict2']['Name']}\n")


def demonstrate_dictionary_deletion():
    """Shows how to delete elements from a dictionary."""
    print("--- Demonstrating Dictionary Deletion ---")

    my_dict = {
        5: 'Welcome', 6: 'To', 7: 'Geeks',
        'A': {1: 'Geeks', 2: 'For', 3: 'Geeks'},
        'B': {1: 'Geeks', 2: 'Life'}
    }
    print(f"Initial dictionary: {my_dict}")

    # 1. Deleting a specific key using 'del'
    del my_dict[6]
    print(f"\nAfter deleting key 6: {my_dict}")

    # 2. Deleting a key from a nested dictionary
    del my_dict['A'][2]
    print(f"\nAfter deleting key 2 from nested dict 'A': {my_dict}")

    # 3. Deleting an arbitrary key-value pair using popitem()
    # In Python 3.7+, popitem() removes the last inserted item (LIFO).
    popped_element = my_dict.popitem()
    print(f"\nDictionary after popitem(): {my_dict}")
    print(f"The arbitrary pair returned is: {popped_element}")

    # 4. Deleting all items using clear()
    my_dict.clear()
    print(f"\nAfter clearing the entire dictionary: {my_dict}\n")


def find_set_difference_in_dict_lists():
    """
    Finds the symmetric difference between two lists of dictionaries.
    (Items that are in one list but not the other).
    """
    print("--- Demonstrating Set Difference on Lists of Dictionaries ---")
    list1 = [{"HpY": 22}, {"BirthdaY": 2}]
    list2 = [{"HpY": 22}, {"BirthdaY": 2}, {"Shambhavi": 2019}]

    print(f"\nOriginal list 1: {list1}")
    print(f"Original list 2: {list2}")

    # Symmetric difference using list comprehensions
    symmetric_diff = [item for item in list1 if item not in list2] + \
                     [item for item in list2 if item not in list1]

    print(f"\nThe set difference of the lists is: {symmetric_diff}\n")


def convert_list_of_dicts_to_nested_dict():
    """
    Converts a list of dictionaries to a single nested dictionary
    where the keys are the list indices.
    """
    print("--- Converting List of Dictionaries to a Nested Dictionary ---")
    test_list = [{"Gfg": 3, 4: 9}, {"is": 8, "Good": 2}, {"Best": 10, "CS": 1}]
    print(f"\nOriginal list: {test_list}")

    # Using a dictionary comprehension (more concise and Pythonic)
    nested_dict = {index: value for index, value in enumerate(test_list)}
    print(f"Constructed dictionary: {nested_dict}\n")


def merge_dict_key_values_to_list():
    """
    Merges a list of dictionaries into a single dictionary where values
    for the same key are collected into a list.
    """
    print("--- Merging Key Values from a List of Dictionaries ---")
    test_list = [
        {'gfg': 2, 'is': 4, 'best': 6},
        {'it': 5, 'is': 7, 'best': 8},
        {'CS': 10}
    ]
    print(f"\nOriginal list: {test_list}")

    merged_dict: Dict[Any, List] = {}
    for sub_dict in test_list:
        for key, value in sub_dict.items():
            # setdefault gets the key's value, or sets it to [] if not present
            merged_dict.setdefault(key, []).append(value)

    print(f"The merged dictionary is: {merged_dict}\n")


def main():
    """Run all dictionary demonstration functions."""
    demonstrate_dictionary_creation()
    demonstrate_dictionary_access()
    demonstrate_dictionary_deletion()
    find_set_difference_in_dict_lists()
    convert_list_of_dicts_to_nested_dict()
    merge_dict_key_values_to_list()


if __name__ == "__main__":
    main()




"""
list_and_tuple_examples.py

A collection of examples demonstrating operations on lists and tuples,
including sorting, data conversion, and comprehensions.
"""
from typing import Any, Dict, List, Tuple


def sort_list_of_dicts_by_value():
    """Sorts a list of dictionaries based on a specific key's value."""
    print("--- Sorting a List of Dictionaries by a Key's Value ---")
    
    # Example 1: Sorting by a string date
    birthdays = [
        {'name': 'akash', 'd.o.b': '1997-03-02'},
        {'name': 'manjeet', 'd.o.b': '1997-01-04'},
        {'name': 'nikhil', 'd.o.b': '1997-09-13'}
    ]
    print(f"\nInitial list: {birthdays}")
    birthdays.sort(key=lambda item: item['d.o.b'])
    print(f"Sorted by date of birth: {birthdays}")

    # Example 2: Sorting by age (numeric)
    people = [
        {"name": "Nandini", "age": 20},
        {"name": "Manjeet", "age": 20},
        {"name": "Nikhil", "age": 19}
    ]
    print(f"\nInitial list: {people}")
    sorted_people = sorted(people, key=lambda item: item['age'])
    print(f"Sorted by age: {sorted_people}")
    
    # Example 3: Sorting by multiple criteria (age, then name)
    # The `sorted` function is stable, so we can sort sequentially.
    sorted_by_name = sorted(people, key=lambda item: item['name'])
    sorted_by_age_then_name = sorted(sorted_by_name, key=lambda item: item['age'])
    # Alternatively, use a tuple in the key
    # sorted_by_age_then_name_alt = sorted(people, key=lambda item: (item['age'], item['name']))
    print(f"Sorted by age, then name: {sorted_by_age_then_name}\n")


def convert_list_of_dicts_to_list_of_lists():
    """
    Converts a list of dictionaries into a list of lists,
    with the first inner list being the headers (keys).
    """
    print("--- Converting a List of Dictionaries to a List of Lists ---")
    data = [
        {'Nikhil': 17, 'Akash': 18, 'Akshat': 20},
        {'Nikhil': 21, 'Akash': 30, 'Akshat': 10},
        {'Nikhil': 31, 'Akash': 12, 'Akshat': 19}
    ]
    print(f"\nOriginal list of dictionaries: {data}")

    if not data:
        print("Converted list: []")
        return

    # The header is the list of keys from the first dictionary
    header = list(data[0].keys())
    # The rows are the lists of values from each dictionary
    rows = [list(item.values()) for item in data]
    
    result = [header] + rows
    print(f"The converted list of lists: {result}\n")


def filter_tuples_from_list():
    """Removes tuples from a list based on a condition."""
    print("--- Filtering Tuples from a List ---")
    
    # 1. Using a list comprehension (most common)
    initial_tuples = [('b', 100), ('c', 200), ('c', 45), ('d', 876), ('e', 75)]
    print(f"\nInitial list of tuples: {initial_tuples}")
    
    # Condition: keep tuples where the second element is <= 100
    filtered_tuples = [t for t in initial_tuples if t[1] <= 100]
    print(f"Resultant list (comprehension): {filtered_tuples}")

    # 2. Using the filter() function
    filtered_with_func = list(filter(lambda t: t[1] <= 100, initial_tuples))
    print(f"Resultant list (filter function): {filtered_with_func}\n")


def extract_numeric_string_tuples():
    """Extracts tuples from a list where all elements are numeric strings."""
    print("--- Extracting Tuples with All Numeric Strings ---")
    test_list = [("45", "86"), ("Gfg", "1"), ("98", "10"), ("Gfg", "Best")]
    print(f"\nThe original list is: {test_list}")

    # Use a list comprehension with all() and str.isdigit()
    result = [
        sub_tuple for sub_tuple in test_list 
        if all(element.isdigit() for element in sub_tuple)
    ]
    print(f"Filtered Tuples: {result}\n")


def main():
    """Run all list and tuple demonstration functions."""
    sort_list_of_dicts_by_value()
    convert_list_of_dicts_to_list_of_lists()
    filter_tuples_from_list()
    extract_numeric_string_tuples()


if __name__ == "__main__":
    main()






"""
functions_and_algorithms.py

A collection of fundamental functions and algorithms, refactored for clarity,
reusability, and adherence to Python best practices.
"""
import math
from typing import List, Tuple


def is_prime(number: int) -> bool:
    """
    Checks if a number is a prime number.

    A prime number is a natural number greater than 1 that has no positive
    divisors other than 1 and itself.

    Args:
        number: The integer to check.

    Returns:
        True if the number is prime, False otherwise.
    """
    if number <= 1:
        return False
    # Check for divisors from 2 up to the square root of the number
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False
    return True


def demonstrate_swapping_numbers():
    """Shows two ways to swap the values of two variables."""
    print("--- Demonstrating Swapping Numbers ---")
    num1, num2 = 5, 10
    print(f"Original values: num1 = {num1}, num2 = {num2}")

    # The Pythonic way using tuple unpacking
    num1, num2 = num2, num1
    print(f"After Pythonic swap: num1 = {num1}, num2 = {num2}")

    # The traditional way using a temporary variable
    num1, num2 = 5, 10 # Reset
    temp = num1
    num1 = num2
    num2 = temp
    print(f"After traditional swap: num1 = {num1}, num2 = {num2}\n")


def factorial(n: int) -> int:
    """
    Calculates the factorial of a non-negative integer.

    Args:
        n: The non-negative integer.

    Returns:
        The factorial of n.
    
    Raises:
        ValueError: If n is a negative number.
    """
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if n == 0:
        return 1
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result


def find_largest_element(arr: List[int]) -> int:
    """
    Finds the largest element in a list of numbers.

    Args:
        arr: A list of integers.

    Returns:
        The largest integer in the list.

    Raises:
        ValueError: If the list is empty.
    """
    if not arr:
        raise ValueError("Input list cannot be empty.")
    # The built-in max() function is the most Pythonic way
    return max(arr)


def is_palindrome(text: str) -> bool:
    """
    Checks if a string is a palindrome.

    A palindrome is a word, phrase, or sequence that reads the same
    backwards as forwards. This check is case-sensitive.

    Args:
        text: The string to check.

    Returns:
        True if the string is a palindrome, False otherwise.
    """
    return text == text[::-1]


def kilometers_to_miles(kilometers: float) -> float:
    """Converts a distance from kilometers to miles."""
    CONVERSION_FACTOR = 0.621371
    return kilometers * CONVERSION_FACTOR


def get_character_ascii_value(char: str) -> int:
    """
    Returns the ASCII value of a single character.

    Args:
        char: A string of length 1.

    Returns:
        The integer ASCII value.
    """
    if len(char) != 1:
        raise ValueError("Input must be a single character.")
    return ord(char)


def find_divisible_by_7_not_5(start: int, end: int) -> List[int]:
    """
    Finds all numbers in a range divisible by 7 but not by 5.

    Args:
        start: The starting number of the range (inclusive).
        end: The ending number of the range (inclusive).

    Returns:
        A list of integers that meet the criteria.
    """
    return [
        num for num in range(start, end + 1)
        if num % 7 == 0 and num % 5 != 0
    ]


def main():
    """Run all function and algorithm demonstrations."""
    # is_prime demonstration
    print("--- Prime Number Check ---")
    print(f"Is 7 prime? {is_prime(7)}")
    print(f"Is 10 prime? {is_prime(10)}\n")

    demonstrate_swapping_numbers()

    # factorial demonstration
    print("--- Factorial Calculation ---")
    print(f"The factorial of 5 is: {factorial(5)}\n")

    # find_largest_element demonstration
    print("--- Find Largest Element in a List ---")
    print(f"Largest in [1, 20, 3, 50, 12] is: {find_largest_element([1, 20, 3, 50, 12])}\n")

    # is_palindrome demonstration
    print("--- Palindrome Check ---")
    print(f"Is 'malayalam' a palindrome? {is_palindrome('malayalam')}")
    print(f"Is 'python' a palindrome? {is_palindrome('python')}\n")

    # find_divisible_by_7_not_5 demonstration
    print("--- Find numbers divisible by 7 but not 5 (2000-3200) ---")
    result_nums = find_divisible_by_7_not_5(2000, 3200)
    print(f"Found {len(result_nums)} numbers. First 10: {result_nums[:10]}\n")


if __name__ == "__main__":
    main()







"""
oop_and_exceptions.py

Examples demonstrating basic Object-Oriented Programming (OOP) concepts
like classes, inheritance, and methods, as well as exception handling.
"""
import math


class Shape:
    """A base class for geometric shapes."""
    def area(self) -> float:
        """
        Calculates the area of the shape.
        The base shape has an area of 0.
        """
        return 0.0


class Circle(Shape):
    """A class representing a circle, inheriting from Shape."""
    def __init__(self, radius: float):
        """
        Initializes a Circle with a given radius.

        Args:
            radius: The radius of the circle.
        """
        if radius < 0:
            raise ValueError("Radius cannot be negative.")
        self.radius = radius

    def area(self) -> float:
        """Calculates the area of the circle (π * r^2)."""
        return math.pi * (self.radius ** 2)


class Rectangle(Shape):
    """A class representing a rectangle, inheriting from Shape."""
    def __init__(self, length: float, width: float):
        """Initializes a Rectangle with a length and width."""
        if length < 0 or width < 0:
            raise ValueError("Length and width cannot be negative.")
        self.length = length
        self.width = width

    def area(self) -> float:
        """Calculates the area of the rectangle (length * width)."""
        return self.length * self.width


def demonstrate_shapes():
    """Creates instances of shapes and prints their areas."""
    print("--- Demonstrating OOP with Shapes ---")
    my_circle = Circle(radius=10)
    my_rectangle = Rectangle(length=4, width=5)

    print(f"The radius of the circle is {my_circle.radius}")
    print(f"The area of the circle is: {my_circle.area():.2f}")
    print(f"\nThe length and width of the rectangle are {my_rectangle.length} and {my_rectangle.width}")
    print(f"The area of the rectangle is: {my_rectangle.area():.2f}\n")


# --- Exception Handling ---

class CustomValidationError(Exception):
    """A custom exception for validation errors."""
    def __init__(self, message: str, field: str):
        super().__init__(message)
        self.field = field

    def __str__(self) -> str:
        return f"Validation Error on field '{self.field}': {self.args[0]}"


def demonstrate_exception_handling():
    """Shows how to use try/except/finally and custom exceptions."""
    print("--- Demonstrating Exception Handling ---")
    
    # 1. Catching a built-in exception
    try:
        result = 5 / 0
    except ZeroDivisionError as e:
        print(f"Caught an expected error: {e}")
    finally:
        print("This 'finally' block always executes, error or not.")

    # 2. Raising and catching a custom exception
    try:
        user_email = "not-an-email"
        if "@" not in user_email:
            raise CustomValidationError("Email address is invalid", field="email")
    except CustomValidationError as e:
        print(f"\nCaught a custom error: {e}")


def main():
    """Run all OOP and exception handling demonstrations."""
    demonstrate_shapes()
    demonstrate_exception_handling()


if __name__ == "__main__":
    main()
