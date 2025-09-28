# -*- coding: utf-8 -*-
"""
A collection of refactored Python code examples demonstrating best practices.

This script covers various topics including data structures, algorithms,
threading, decorators, and common programming tasks. Each example is
encapsulated in a well-documented function or class.
"""

import threading
import random
import re
import secrets
import string
import requests
from functools import reduce

# ==============================================================================
# 1. Data Structures and Collections
# ==============================================================================

def transpose_matrix(matrix: list[list]) -> list[list]:
    """
    Transposes a matrix (swaps rows and columns).

    Args:
        matrix: A list of lists representing the matrix.

    Returns:
        The transposed matrix.
    """
    if not matrix or not matrix[0]:
        return []
    # Use a list comprehension for a concise and efficient solution.
    num_cols = len(matrix[0])
    return [[row[i] for row in matrix] for i in range(num_cols)]

def reverse_dict_order(d: dict) -> dict:
    """
    Reverses the key order of a dictionary (requires Python 3.7+).

    Args:
        d: The input dictionary.

    Returns:
        A new dictionary with the key order reversed.
    """
    return dict(reversed(list(d.items())))

def invert_dictionary(d: dict) -> dict:
    """
    Inverts a dictionary, swapping keys and values.

    Args:
        d: The input dictionary. Assumes values are unique and hashable.

    Returns:
        A new dictionary with keys and values swapped.
    """
    return {value: key for key, value in d.items()}

def remove_duplicates_from_list(items: list) -> list:
    """
    Removes duplicate items from a list while preserving original order.

    Args:
        items: A list that may contain duplicate elements.

    Returns:
        A new list with unique elements in their original order.
    """
    # dict.fromkeys() is an efficient and order-preserving way to get unique items.
    return list(dict.fromkeys(items))

def get_unique_values_from_tuples(tuples: list[tuple]) -> list:
    """
    Gets unique values from a list of tuples.

    Args:
        tuples: A list of tuples.

    Returns:
        A list of unique tuples.
    """
    return list(set(tuples))

def sort_tuples_by_second_item(tuples: list[tuple]) -> list[tuple]:
    """
    Sorts a list of tuples based on the second element of each tuple.

    Args:
        tuples: A list of tuples.

    Returns:
        A new list of tuples sorted by the second item.
    """
    # Use the `key` argument of sorted() for a clean and efficient sort.
    return sorted(tuples, key=lambda item: item[1])

def find_common_elements(list1: list, list2: list) -> list:
    """
    Finds common elements between two lists.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        A list of elements present in both lists.
    """
    return list(set(list1) & set(list2))

def create_dict_from_lists(keys: list, values: list) -> dict:
    """
    Creates a dictionary from two lists of keys and values.

    Args:
        keys: A list of keys.
        values: A list of values.

    Returns:
        A dictionary created from the keys and values.
    """
    return dict(zip(keys, values))

# ==============================================================================
# 2. Algorithms and Numeric Operations
# ==============================================================================

def generate_fibonacci_sequence(n_terms: int):
    """
    A generator function that yields Fibonacci numbers.

    Args:
        n_terms: The number of Fibonacci terms to generate.
    """
    a, b = 0, 1
    for _ in range(n_terms):
        yield a
        a, b = b, a + b

def greatest_common_divisor(a: int, b: int) -> int:
    """
    Calculates the greatest common divisor (GCD) of two integers using the
    Euclidean algorithm.

    Args:
        a: The first integer.
        b: The second integer.

    Returns:
        The greatest common divisor of a and b.
    """
    while b:
        a, b = b, a % b
    return a

def reverse_integer(n: int) -> int:
    """
    Reverses the digits of an integer.

    Args:
        n: The integer to reverse.

    Returns:
        The reversed integer.
    """
    reversed_num = 0
    is_negative = n < 0
    n = abs(n)
    while n > 0:
        digit = n % 10
        reversed_num = reversed_num * 10 + digit
        n //= 10
    return -reversed_num if is_negative else reversed_num

def sum_even_numbers(numbers: list[int]) -> int:
    """
    Calculates the sum of only the even numbers in a list.

    Args:
        numbers: A list of integers.

    Returns:
        The sum of the even numbers.
    """
    # A generator expression with sum() is highly readable and efficient.
    return sum(num for num in numbers if num % 2 == 0)

def sum_every_nth_element(numbers: list, n: int) -> int:
    """
    Adds every n-th element in a list.

    Args:
        numbers: A list of numbers.
        n: The interval of elements to sum (e.g., 3 for every 3rd).

    Returns:
        The sum of every n-th element.
    """
    # List slicing is the most Pythonic way to solve this.
    return sum(numbers[n-1::n])

def calculate_average(scores: list[float]) -> float:
    """
    Calculates the average of a list of scores.

    Args:
        scores: A list of numbers.

    Returns:
        The average of the scores, or 0 if the list is empty.
    """
    if not scores:
        return 0.0
    return sum(scores) / len(scores)

# ==============================================================================
# 3. String Manipulation
# ==============================================================================

def strip_vowels(text: str) -> str:
    """
    Removes all vowels from a string.

    Args:
        text: The input string.

    Returns:
        The string with all vowels (a, e, i, o, u) removed.
    """
    vowels = "aeiou"
    return "".join([char for char in text if char.lower() not in vowels])

def shift_characters(text: str, shift: int) -> str:
    """
    Shifts each letter in a string by a given number of places (Caesar cipher).
    Wraps around the alphabet and preserves case.

    Args:
        text: The string to process.
        shift: The number of letters to shift by.

    Returns:
        The string with characters shifted.
    """
    result = []
    for char in text:
        if 'a' <= char <= 'z':
            start = ord('a')
            new_ord = (ord(char) - start + shift) % 26 + start
            result.append(chr(new_ord))
        elif 'A' <= char <= 'Z':
            start = ord('A')
            new_ord = (ord(char) - start + shift) % 26 + start
            result.append(chr(new_ord))
        else:
            result.append(char) # Append non-alphabetic characters as is
    return "".join(result)

def count_char_types(text: str) -> dict:
    """
    Counts the number of characters, digits, and symbols in a string.

    Args:
        text: The input string.

    Returns:
        A dictionary with counts for 'chars', 'digits', and 'symbols'.
    """
    counts = {'chars': 0, 'digits': 0, 'symbols': 0}
    for char in text:
        if char.isalpha():
            counts['chars'] += 1
        elif char.isdigit():
            counts['digits'] += 1
        else:
            counts['symbols'] += 1
    return counts

def replace_punctuation(text: str, replacement: str = '#') -> str:
    """
    Replaces all punctuation characters in a string with a replacement character.

    Args:
        text: The input string.
        replacement: The character to replace punctuation with. Defaults to '#'.

    Returns:
        The string with punctuation replaced.
    """
    translator = str.maketrans(string.punctuation, replacement * len(string.punctuation))
    return text.translate(translator)

# ==============================================================================
# 4. Utilities and Miscellaneous
# ==============================================================================

def type_conversion(value, target_type: str):
    """
    Converts a given variable to a specified data type ('int', 'float', 'str').

    Args:
        value: The value to convert.
        target_type: The name of the type to convert to.

    Returns:
        The converted value.
    """
    type_map = {'int': int, 'float': float, 'str': str}
    converter = type_map.get(target_type)
    if converter:
        return converter(value)
    raise ValueError(f"Unsupported type for conversion: {target_type}")

def celsius_to_fahrenheit(celsius: float) -> float:
    """Converts temperature from Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

def kph_to_mph(kph: float) -> float:
    """Converts speed from kilometers per hour to miles per hour."""
    KM_TO_MILES_RATIO = 0.621371
    return kph * KM_TO_MILES_RATIO

def get_external_ip() -> str | None:
    """
    Fetches the external IP address from an online service.

    Returns:
        The external IP address as a string, or None if an error occurs.
    """
    try:
        response = requests.get("http://checkip.dyndns.org", timeout=5)
        response.raise_for_status()
        # Use regex to reliably find the IP address
        ip_match = re.search(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", response.text)
        return ip_match.group(0) if ip_match else None
    except requests.exceptions.RequestException as e:
        print(f"Error fetching IP address: {e}")
        return None

def generate_secure_password(length: int = 12) -> str:
    """
    Generates a cryptographically secure random password.

    Args:
        length: The desired length of the password. Must be at least 4.

    Returns:
        A secure, random password.
    """
    if length < 4:
        raise ValueError("Password length must be at least 4 to include all character types.")
        
    alphabet = string.ascii_letters + string.digits + string.punctuation
    
    while True:
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        # Ensure the password contains at least one of each character type
        if (any(c.islower() for c in password)
                and any(c.isupper() for c in password)
                and any(c.isdigit() for c in password)
                and any(c in string.punctuation for c in password)):
            return password

# ==============================================================================
# 5. Classes, Decorators, and Threading
# ==============================================================================

class Greeter:
    """A simple class to greet a person by name."""
    def __init__(self, name: str):
        """Initializes the Greeter with a name."""
        self.name = name

    def greet(self, uppercase: bool = False):
        """
        Prints a greeting message.

        Args:
            uppercase: If True, prints the name in uppercase.
        """
        name_to_print = self.name.upper() if uppercase else self.name
        print(f"Hello, {name_to_print}!")

def add_currency_symbol(func):
    """Decorator that adds a '$' symbol to the result of a function."""
    def wrapper(amount):
        original_result = func(amount)
        return f"${original_result}"
    return wrapper

@add_currency_symbol
def format_price(price: str) -> str:
    """Formats a price string."""
    return price

def run_threaded_task(target_func, args_tuple: tuple):
    """
    Creates, starts, and joins a thread for a given function.
    
    Args:
        target_func: The function to be executed in the thread.
        args_tuple: A tuple of arguments for the target function.
    """
    print(f"Starting thread for {target_func.__name__} with args {args_tuple}")
    thread = threading.Thread(target=target_func, args=args_tuple)
    thread.start()
    thread.join()
    print("Thread finished.")

def print_cube(num: int):
    """Prints the cube of a number."""
    print(f"The cube of {num} is {num ** 3}")


# ==============================================================================
# 6. Interactive Scripts (Wrapped in functions)
# ==============================================================================

def authenticate_user():
    """Prompts for a username and checks against a list of allowed users."""
    allowed_users = {"Ram", "Mohan"} # Use a set for efficient lookups
    username = input("Enter your username: ")
    if username in allowed_users:
        print("Access granted. Welcome!")
    else:
        print("Access denied.")

def play_dice_game():
    """A simple interactive dice rolling game."""
    while True:
        print("Rolling the dice...")
        print(f"Dice 1: {random.randint(1, 6)}")
        print(f"Dice 2: {random.randint(1, 6)}")

        roll_again = input("Roll again? (yes/no): ").lower()
        if roll_again not in ("yes", "y"):
            print("Thanks for playing!")
            break

# ==============================================================================
# Main Execution Block
# ==============================================================================

def main():
    """Main function to demonstrate the refactored code."""
    print("--- Demonstrating Refactored Python Code ---\n")

    # 1. Matrix Transposition
    print("1. Matrix Transposition")
    matrix = [[1, 2, 3], [4, 5, 6]]
    transposed = transpose_matrix(matrix)
    print(f"Original matrix: {matrix}")
    print(f"Transposed matrix: {transposed}\n")

    # 2. Threading Example
    print("2. Threading Demonstration")
    run_threaded_task(target_func=print_cube, args_tuple=(10,))
    print("")

    # 3. Decorator Example
    print("3. Decorator Demonstration")
    print(f"Formatted price: {format_price('100.50')}\n")

    # 4. Type Conversion
    print("4. Type Conversion")
    print(f"Converting '123' to int: {type_conversion('123', 'int')}")
    print(f"Converting 1 to str: {type_conversion(1, 'str')}\n")

    # 5. Class Example
    print("5. Class Demonstration")
    greeter = Greeter("Geeta")
    greeter.greet()
    greeter.greet(uppercase=True)
    print("")

    # 6. Dictionary Operations
    print("6. Dictionary Operations")
    sample_dict = {1: 'Hi', 2: 'Hello', 3: 'Hey'}
    print(f"Original dictionary: {sample_dict}")
    print(f"Reversed order: {reverse_dict_order(sample_dict)}")
    print(f"Inverted (key/value swap): {invert_dictionary(sample_dict)}\n")

    # 7. Fibonacci Sequence
    print("7. Fibonacci Sequence")
    fib_sequence = list(generate_fibonacci_sequence(10))
    print(f"First 10 Fibonacci numbers: {fib_sequence}\n")

    # 8. Sum of even numbers and every Nth element
    print("8. List Summation Operations")
    numbers = list(range(1, 21)) # Numbers from 1 to 20
    print(f"List: {numbers}")
    print(f"Sum of even numbers: {sum_even_numbers(numbers)}")
    print(f"Sum of every 5th element: {sum_every_nth_element(numbers, 5)}\n")

    # 9. String Operations
    print("9. String Operations")
    print(f"Stripping vowels from 'hello world': '{strip_vowels('hello world')}'")
    print(f"Caesar cipher on 'hello' with shift 5: '{shift_characters('hello', 5)}'")
    print(f"Replacing punctuation in 'Hi! How are you?': '{replace_punctuation('Hi! How are you?')}'\n")

    # 10. Secure Password Generation
    print("10. Secure Password Generation")
    print(f"Generated password: {generate_secure_password(16)}\n")
    
    # 11. Get External IP
    print("11. Get External IP Address")
    ip = get_external_ip()
    if ip:
        print(f"Your external IP is: {ip}\n")
    else:
        print("Could not retrieve external IP.\n")

    # 12. Interactive examples (uncomment to run)
    # print("12. Interactive Examples")
    # authenticate_user()
    # play_dice_game()

if __name__ == "__main__":
    main()


transposed = []
matrix = [[1, 2, 3, 4], [4, 5, 6, 8]]

for i in range(len(matrix[0])):
    transposed_row = []
    for row in matrix:
        transposed_row.append(row[i])
    transposed.append(transposed_row)
print(f"{transposed}")
```

**Refactored Logic (in `transpose_matrix` function):**
```python
def transpose_matrix(matrix: list[list]) -> list[list]:
    """Transposes a matrix (swaps rows and columns)."""
    if not matrix or not matrix[0]:
        return []
    num_cols = len(matrix[0])
    return [[row[i] for row in matrix] for i in range(num_cols)]




import threading 
def print_cube(num): 
    print("Cube: {}".format(num * num * num)) 
t1 = threading.Thread(target=print_cube, args=(10,)) 
t2 = threading.Thread(target=print_cube, args=(10,)) 
t1.start() 
t2.start() 
t1.join()
t2.join()
```

**Refactored Logic (in `run_threaded_task` and `main`):**
```python
def run_threaded_task(target_func, args_tuple: tuple):
    """Creates, starts, and joins a thread for a given function."""
    thread = threading.Thread(target=target_func, args=args_tuple)
    thread.start()
    thread.join()

def print_cube(num: int):
    """Prints the cube of a number."""
    print(f"The cube of {num} is {num ** 3}")

# Called from main()
run_threaded_task(target_func=print_cube, args_tuple=(10,))




def myDecorator(func):
    def new_func(n):
        return '$' + func(n)        
    return new_func

@myDecorator
def myFunction(a):
    return(a)
print(myFunction('100'))
```

**Refactored Logic (as `add_currency_symbol`):**
```python
def add_currency_symbol(func):
    """Decorator that adds a '$' symbol to the result of a function."""
    def wrapper(amount):
        original_result = func(amount)
        return f"${original_result}"
    return wrapper

@add_currency_symbol
def format_price(price: str) -> str:
    """Formats a price string."""
    return price




nterms = int(10000)
# ... long while loop ...
fab_list.append(n1)
fn = lambda x: "FIBONACCI" if x in fab_list else "NOT_FIBONACCI"
print("Given number is",fn(20))
```

**Refactored Logic (as `generate_fibonacci_sequence` generator):**
```python
def generate_fibonacci_sequence(n_terms: int):
    """A generator function that yields Fibonacci numbers."""
    a, b = 0, 1
    for _ in range(n_terms):
        yield a
        a, b = b, a + b




from functools import reduce
input_list = [x for x in range(100)]
def sum_even(it):
    return reduce(lambda x, y: x + y if (y % 2)==0 else x, it, 0)


res=reduce((lambda x, y: x + y), [val for idx, val in enumerate(input_list) if (idx+1)%5==0])


def sum_even_numbers(numbers: list[int]) -> int:
    """Calculates the sum of only the even numbers in a list."""
    return sum(num for num in numbers if num % 2 == 0)

def sum_every_nth_element(numbers: list, n: int) -> int:
    """Adds every n-th element in a list."""
    return sum(numbers[n-1::n])


import secrets
import string

def generate_secure_password(length: int = 12) -> str:
    """Generates a cryptographically secure random password."""
    if length < 4:
        raise ValueError("Password length must be at least 4...")
    alphabet = string.ascii_letters + string.digits + string.punctuation
    while True:
        password = ''.join(secrets.choice(alphabet) for _ in range(length))
        if (any(c.islower() for c in password) # and so on...
            # ... check for all required character types
           ):
            return password


def remove_duplicates(lista):
    lista2 = []
    if lista: 
        for item in lista:
            if item not in lista2:
                lista2.append(item)
    # ...
    return lista2


def remove_duplicates_from_list(items: list) -> list:
    """Removes duplicate items from a list while preserving original order."""
    return list(dict.fromkeys(items))
