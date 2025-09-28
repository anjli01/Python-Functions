# -*- coding: utf-8 -*-
"""
A collection of refactored Python code examples demonstrating standard practices.

This script includes functions and snippets for various tasks, including:
- Data structure manipulation (lists, tuples, dictionaries)
- Algorithmic problems (sorting, searching, recursion)
- Mathematical and scientific calculations (geometry, physics)
- Utility functions (file I/O, time conversion)
"""

import math
from typing import (
    Any, Dict, List, Tuple, Union, Sequence, TypeVar, Optional, Callable
)

# Define a generic type for sequences that can be sorted.
SortableItem = TypeVar('SortableItem')
SortableSequence = Union[List[SortableItem], Tuple[SortableItem, ...], str]


# --- Constants ---
# Using constants for values that do not change improves readability and maintainability.
PI = math.pi
GRAVITATIONAL_ACCELERATION = 9.8  # m/s^2
GAS_CONSTANT_R = 8.3145  # J/(mol·K)
SPEED_OF_LIGHT_C = 299_792_458  # m/s, using underscores for readability


# ==============================================================================
# --- Data Structure Operations ---
# ==============================================================================

# --- List and Tuple Operations ---

def filter_positive_tuples(data: List[tuple]) -> List[tuple]:
    """
    Filters a list of tuples, returning only those containing all non-negative elements.

    Args:
        data: A list of tuples, where each tuple contains numbers.

    Returns:
        A new list containing only the tuples with all non-negative elements.
    """
    return [tup for tup in data if all(element >= 0 for element in tup)]


def group_tuples_by_first_element(data: List[Tuple[Any, ...]]) -> List[tuple]:
    """
    Groups tuples from a list if they share the same initial element.

    Example: [(5, 6), (5, 7), (6, 8)] -> [(5, 6, 7), (6, 8)]

    Args:
        data: A list of tuples. Assumes the list is sorted by the first element.

    Returns:
        A list of grouped tuples.
    """
    if not data:
        return []

    result = []
    # Start with the first tuple converted to a list for mutability
    current_group = list(data[0])

    for i in range(1, len(data)):
        current_tuple = data[i]
        if current_tuple[0] == current_group[0]:
            # Extend the current group with the rest of the tuple's elements
            current_group.extend(current_tuple[1:])
        else:
            # Finalize the previous group and start a new one
            result.append(tuple(current_group))
            current_group = list(current_tuple)

    # Append the last processed group
    result.append(tuple(current_group))
    return result


def find_uncommon_elements(list1: List[list], list2: List[list]) -> List[list]:
    """
    Finds elements that are not common between two lists of lists.

    Note: This approach converts inner lists to tuples to make them hashable
    for use in sets, which provides an efficient way to find the symmetric
    difference.

    Args:
        list1: The first list of lists.
        list2: The second list of lists.

    Returns:
        A list containing the uncommon lists.
    """
    # Sets require hashable types, so we convert inner lists to tuples.
    set1 = {tuple(item) for item in list1}
    set2 = {tuple(item) for item in list2}

    # The symmetric difference gives elements that are in one set, but not both.
    uncommon_tuples = set1.symmetric_difference(set2)

    # Convert the resulting tuples back to lists.
    return [list(item) for item in uncommon_tuples]


def shift_and_scale(
    numbers: List[float], mean: float, std_dev: float
) -> List[float]:
    """
    Shifts and scales all numbers in a list using a given mean and standard deviation.

    This process is also known as standardization or calculating z-scores.

    Args:
        numbers: A list of numbers to transform.
        mean: The mean to subtract from each number.
        std_dev: The standard deviation to divide each number by.

    Returns:
        A new list of the transformed numbers.

    Raises:
        ValueError: If the standard deviation is zero.
    """
    if std_dev == 0:
        raise ValueError("Standard deviation cannot be zero.")
    return [(x - mean) / std_dev for x in numbers]


def zip_sequences(list_of_sequences: List[Sequence]) -> List[tuple]:
    """
    Zips corresponding elements from a list of sequences into tuples.

    This is a wrapper around the built-in zip function with argument unpacking.

    Args:
        list_of_sequences: A list where each element is a sequence (e.g., list, tuple).

    Returns:
        A list of tuples, where each tuple contains the i-th element
        from each of the input sequences.
    """
    return list(zip(*list_of_sequences))


# --- Dictionary Operations ---

def merge_dictionaries(dict1: dict, dict2: dict) -> dict:
    """
    Merges two dictionaries. The second dictionary's items overwrite the first's
    in case of key conflicts.

    Args:
        dict1: The first dictionary.
        dict2: The second dictionary, whose items will be added to the first.

    Returns:
        A new dictionary containing items from both input dictionaries.
    """
    # Using the spread operator {**dict1, **dict2} is a modern, clean way
    # to merge dictionaries without modifying the originals.
    return {**dict1, **dict2}


def filter_by_key_value(
    list_of_dicts: List[Dict[str, Any]], key: str, value: Any
) -> List[Dict[str, Any]]:
    """
    Filters a list of dictionaries, returning only those where a specified
    key matches a given value.

    Args:
        list_of_dicts: The list of dictionaries to filter.
        key: The key to check in each dictionary.
        value: The value to match against.

    Returns:
        A new list of dictionaries that match the criteria.
    """
    # Using a list comprehension is often more readable than filter() with lambda.
    return [d for d in list_of_dicts if d.get(key) == value]


# ==============================================================================
# --- String and I/O Operations ---
# ==============================================================================

def convert_12h_to_24h(time_12h: str) -> str:
    """
    Converts a time string from 12-hour format (e.g., "07:05:45 PM")
    to 24-hour format (e.g., "19:05:45").

    Refactored to use the `datetime` module for robustness and reliability.

    Args:
        time_12h: The time string in 12-hour format with AM/PM.

    Returns:
        The time string in 24-hour format.
    """
    from datetime import datetime
    try:
        # Parse the 12-hour format time string
        time_obj = datetime.strptime(time_12h, "%I:%M:%S %p")
        # Format it into a 24-hour format time string
        return time_obj.strftime("%H:%M:%S")
    except ValueError:
        # Handle cases with slightly different formats if needed, e.g., without seconds
        time_obj = datetime.strptime(time_12h, "%I:%M %p")
        return time_obj.strftime("%H:%M")


def read_and_print_file(filepath: str) -> None:
    """
    Reads the entire content of a file and prints it to the console.

    Args:
        filepath: The path to the file to be read.

    Raises:
        FileNotFoundError: If the file does not exist.
        IOError: If there is an error reading the file.
    """
    try:
        with open(filepath, "r", encoding="utf-8") as file:
            print(file.read())
    except FileNotFoundError:
        print(f"Error: The file at '{filepath}' was not found.")
    except IOError as e:
        print(f"Error reading file at '{filepath}': {e}")


def reverse_string(text: str) -> str:
    """
    Reverses a given string.

    Args:
        text: The string to be reversed.

    Returns:
        The reversed string.
    """
    return text[::-1]


def get_ascii_value(character: str) -> int:
    """
    Returns the ASCII value of a given character.

    Args:
        character: A single character string.

    Returns:
        The integer ASCII value.

    Raises:
        TypeError: If the input is not a string of length 1.
    """
    if not isinstance(character, str) or len(character) != 1:
        raise TypeError("Input must be a single character string.")
    return ord(character)


# ==============================================================================
# --- Core Algorithms and Logic ---
# ==============================================================================

def find_largest_and_smallest(numbers: List[float]) -> Optional[Tuple[float, float]]:
    """
    Finds the largest and smallest numbers in a list.

    Args:
        numbers: A list of numbers.

    Returns:
        A tuple containing (max_value, min_value), or None if the list is empty.
    """
    if not numbers:
        return None
    return max(numbers), min(numbers)


def reverse_sequence_recursively(seq: Sequence) -> Sequence:
    """
    Reverses a sequence (list, tuple, or string) using recursion.

    Args:
        seq: The sequence to reverse.

    Returns:
        A new sequence of the same type in reversed order.
    """
    if not seq:
        return seq  # Return an empty sequence of the same type
    # Recursive step: reverse the rest of the sequence and append the first element
    return reverse_sequence_recursively(seq[1:]) + seq[0:1]


def selection_sort(data: SortableSequence) -> SortableSequence:
    """
    Sorts a sequence using the selection sort algorithm.

    Args:
        data: A list, tuple, or string to be sorted.

    Returns:
        A new, sorted list.
    """
    # Work on a mutable copy
    sorted_list = list(data)
    n = len(sorted_list)

    for i in range(n):
        # Find the index of the minimum element in the unsorted part
        min_index = i
        for j in range(i + 1, n):
            if sorted_list[j] < sorted_list[min_index]:
                min_index = j

        # Swap the found minimum element with the first element of the unsorted part
        sorted_list[i], sorted_list[min_index] = sorted_list[min_index], sorted_list[i]

    # Return as the original type if it was a tuple or string
    if isinstance(data, tuple):
        return tuple(sorted_list)
    if isinstance(data, str):
        return "".join(sorted_list)
    return sorted_list


def merge_sort(data: List[SortableItem]) -> List[SortableItem]:
    """
    Sorts a list using the merge sort algorithm (recursively).

    Args:
        data: A list of sortable items.

    Returns:
        A new, sorted list.
    """
    if len(data) <= 1:
        return data

    # 1. Divide
    mid = len(data) // 2
    left_half = data[:mid]
    right_half = data[mid:]

    # 2. Conquer (recursively sort sub-lists)
    sorted_left = merge_sort(left_half)
    sorted_right = merge_sort(right_half)

    # 3. Combine (merge the sorted sub-lists)
    return _merge(sorted_left, sorted_right)

def _merge(
    left: List[SortableItem], right: List[SortableItem]
) -> List[SortableItem]:
    """Helper function for merge_sort to merge two sorted lists."""
    merged = []
    left_idx, right_idx = 0, 0

    while left_idx < len(left) and right_idx < len(right):
        if left[left_idx] < right[right_idx]:
            merged.append(left[left_idx])
            left_idx += 1
        else:
            merged.append(right[right_idx])
            right_idx += 1

    # Append any remaining elements
    merged.extend(left[left_idx:])
    merged.extend(right[right_idx:])
    return merged


def number_guessing_game() -> None:
    """
    An interactive game where the computer guesses a number between 1 and 5.

    The logic is a simple binary search implemented through user prompts.
    """
    print("Think of a number between 1 and 5. I will guess it in 3 chances!")
    print("Please answer with 'y' for yes or 'n' for no.")

    def get_user_confirmation(prompt: str) -> bool:
        """Helper to get a valid y/n answer from the user."""
        while True:
            answer = input(prompt).lower()
            if answer in ('y', 'yes'):
                return True
            if answer in ('n', 'no'):
                return False
            print("Invalid input. Please enter 'y' or 'n'.")

    # The logic follows a decision tree
    if get_user_confirmation("Is your number <= 3? (y/n) "):
        if get_user_confirmation("Is your number <= 1? (y/n) "):
            if get_user_confirmation("Is your number 1? (y/n) "):
                print("Yay! Your number is 1.")
            else:
                # This branch is logically impossible if answers are consistent
                # but we handle it for completeness.
                print("It seems there was a contradiction in your answers.")
        else: # Number is 2 or 3
            if get_user_confirmation("Is your number 2? (y/n) "):
                print("Yay! Your number is 2.")
            else:
                print("Yay! Your number is 3.")
    else: # Number is 4 or 5
        if get_user_confirmation("Is your number 4? (y/n) "):
            print("Yay! Your number is 4.")
        else:
            print("Yay! Your number is 5.")


# ==============================================================================
# --- Mathematical Functions ---
# ==============================================================================

def multiply_three_numbers(num1: float, num2: float, num3: float) -> float:
    """Multiplies three numbers together."""
    return num1 * num2 * num3


def divide_with_remainder(
    dividend: int, divisor: int
) -> Tuple[int, int]:
    """
    Divides the first number by the second and returns the quotient and remainder.

    Args:
        dividend: The number to be divided.
        divisor: The number to divide by.

    Returns:
        A tuple containing (quotient, remainder).

    Raises:
        ZeroDivisionError: If the divisor is 0.
    """
    if divisor == 0:
        raise ZeroDivisionError("Cannot divide by zero.")
    return divmod(dividend, divisor)


def sum_of_first_n_naturals(n: int) -> int:
    """
    Calculates the sum of the first n natural numbers using the formula.

    Args:
        n: A non-negative integer.

    Returns:
        The sum of natural numbers from 1 to n.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    return (n * (n + 1)) // 2


def sum_of_first_n_recursively(n: int) -> int:
    """
    Calculates the sum of the first n natural numbers using recursion.

    Args:
        n: A non-negative integer.

    Returns:
        The sum of natural numbers from 1 to n.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")
    if n == 0:
        return 0
    return n + sum_of_first_n_recursively(n - 1)


def square(number: float) -> float:
    """Calculates the square of a number."""
    return number ** 2


def calculate_hcf(num1: int, num2: int) -> int:
    """
    Calculates the Highest Common Factor (HCF) or Greatest Common Divisor (GCD)
    of two integers using the Euclidean algorithm.

    Args:
        num1: The first integer.
        num2: The second integer.

    Returns:
        The HCF of the two numbers.
    """
    while num2:
        num1, num2 = num2, num1 % num2
    return abs(num1)


def calculate_lcm(num1: int, num2: int) -> int:
    """
    Calculates the Lowest Common Multiple (LCM) of two integers.

    Args:
        num1: The first integer.
        num2: The second integer.

    Returns:
        The LCM of the two numbers.
    """
    if num1 == 0 or num2 == 0:
        return 0
    return abs(num1 * num2) // calculate_hcf(num1, num2)


def find_integer_square_root(number: int) -> Optional[int]:
    """
    Finds the integer square root of a number.

    Args:
        number: A non-negative integer.

    Returns:
        The positive integer square root, or None if it's not a perfect square.
    """
    if number < 0:
        return None
    root = int(math.sqrt(number))
    if root * root == number:
        return root
    return None


# ==============================================================================
# --- Geometry and Physics Formulas ---
# ==============================================================================

# --- 2D Geometry ---

def area_triangle_herons(a: float, b: float, c: float) -> Optional[float]:
    """Calculates the area of a triangle using Heron's formula."""
    if not (a > 0 and b > 0 and c > 0 and (a + b > c) and (a + c > b) and (b + c > a)):
        return None  # Invalid triangle
    s = (a + b + c) / 2
    area = math.sqrt(s * (s - a) * (s - b) * (s - c))
    return area

def area_equilateral_triangle(side: float) -> float:
    """Calculates the area of an equilateral triangle."""
    if side < 0: raise ValueError("Side length must be non-negative.")
    return (math.sqrt(3) / 4) * (side ** 2)

def area_right_triangle(base: float, height: float) -> float:
    """Calculates the area of a right-angled triangle."""
    if base < 0 or height < 0: raise ValueError("Base and height must be non-negative.")
    return (base * height) / 2

def area_rectangle(length: float, width: float) -> float:
    """Calculates the area of a rectangle."""
    if length < 0 or width < 0: raise ValueError("Dimensions must be non-negative.")
    return length * width

def area_square(side: float) -> float:
    """Calculates the area of a square."""
    return area_rectangle(side, side)

def area_circle(radius: float) -> float:
    """Calculates the area of a circle."""
    if radius < 0: raise ValueError("Radius must be non-negative.")
    return PI * (radius ** 2)

def circumference_circle(radius: float) -> float:
    """Calculates the circumference of a circle."""
    if radius < 0: raise ValueError("Radius must be non-negative.")
    return 2 * PI * radius

# --- 3D Geometry ---

def surface_area_sphere(radius: float) -> float:
    """Calculates the surface area of a sphere."""
    if radius < 0: raise ValueError("Radius must be non-negative.")
    return 4 * PI * (radius ** 2)

def volume_sphere(radius: float) -> float:
    """Calculates the volume of a sphere."""
    if radius < 0: raise ValueError("Radius must be non-negative.")
    return (4 / 3) * PI * (radius ** 3)

def volume_cube(side: float) -> float:
    """Calculates the volume of a cube."""
    if side < 0: raise ValueError("Side length must be non-negative.")
    return side ** 3

def volume_cuboid(length: float, width: float, height: float) -> float:
    """Calculates the volume of a cuboid."""
    if length < 0 or width < 0 or height < 0:
        raise ValueError("Dimensions must be non-negative.")
    return length * width * height

# --- Physics ---

def calculate_speed(distance: float, time: float) -> float:
    """Calculates speed given distance and time."""
    if time <= 0: raise ValueError("Time must be positive.")
    return distance / time

def calculate_kinetic_energy(mass: float, velocity: float) -> float:
    """Calculates kinetic energy."""
    if mass < 0: raise ValueError("Mass must be non-negative.")
    return 0.5 * mass * (velocity ** 2)

def calculate_potential_energy(mass: float, height: float) -> float:
    """Calculates gravitational potential energy near Earth's surface."""
    if mass < 0: raise ValueError("Mass must be non-negative.")
    return mass * GRAVITATIONAL_ACCELERATION * height

def calculate_gravitational_force(m1: float, m2: float, r: float) -> float:
    """Calculates the gravitational force between two masses."""
    G = 6.67430e-11  # More precise Gravitational Constant
    if m1 < 0 or m2 < 0: raise ValueError("Masses must be non-negative.")
    if r <= 0: raise ValueError("Distance must be positive.")
    return (G * m1 * m2) / (r ** 2)

def ideal_gas_pressure(n_moles: float, temp_k: float, volume_m3: float) -> float:
    """Calculates the pressure of an ideal gas (PV=nRT)."""
    if n_moles < 0 or temp_k < 0 or volume_m3 <= 0:
        raise ValueError("Moles, temperature (K), and volume must be positive.")
    return (n_moles * GAS_CONSTANT_R * temp_k) / volume_m3


# ==============================================================================
# --- Main Execution Block ---
# ==============================================================================

def main():
    """
    Main function to demonstrate the usage of the refactored functions.
    """
    print("--- Demonstrating Refactored Python Code ---")

    # --- Data Structure Operations ---
    print("\n--- List and Tuple Operations ---")
    
    # Positive Tuples
    tuple_list = [(4, 5, 9), (-3, 2, 3), (-3, 5, 6), (4, 6)]
    print(f"Original list: {tuple_list}")
    positive_tups = filter_positive_tuples(tuple_list)
    print(f"Tuples with only positive elements: {positive_tups}")

    # Group Tuples
    grouping_list = [(5, 6), (5, 7), (6, 8), (6, 10), (7, 13)]
    print(f"\nOriginal list for grouping: {grouping_list}")
    grouped = group_tuples_by_first_element(grouping_list)
    print(f"Grouped by first element: {grouped}")

    # Uncommon Elements
    list_a = [[1, 2], [3, 4], [5, 6]]
    list_b = [[3, 4], [5, 7], [1, 2]]
    print(f"\nFinding uncommon elements between {list_a} and {list_b}")
    uncommon = find_uncommon_elements(list_a, list_b)
    print(f"Uncommon elements: {uncommon}")

    # --- String and I/O Operations ---
    print("\n--- String and Time Operations ---")

    # Time Conversion
    time_12h_str = "07:05:45 PM"
    print(f"Converting 12h time '{time_12h_str}' to 24h format.")
    time_24h_str = convert_12h_to_24h(time_12h_str)
    print(f"Result: {time_24h_str}")
    
    # Reverse String
    original_str = "Hello World"
    print(f"\nReversing the string: '{original_str}'")
    reversed_str = reverse_string(original_str)
    print(f"Reversed: '{reversed_str}'")

    # --- Core Algorithms and Logic ---
    print("\n--- Algorithms ---")

    # Selection Sort
    unsorted_data = [64, 25, 12, 22, 11]
    print(f"Sorting {unsorted_data} with Selection Sort.")
    sorted_data = selection_sort(unsorted_data)
    print(f"Sorted result: {sorted_data}")
    
    # Merge Sort
    unsorted_data_2 = [38, 27, 43, 3, 9, 82, 10]
    print(f"\nSorting {unsorted_data_2} with Merge Sort.")
    sorted_data_2 = merge_sort(unsorted_data_2)
    print(f"Sorted result: {sorted_data_2}")

    # --- Mathematical Functions ---
    print("\n--- Math Functions ---")

    # HCF and LCM
    num1, num2 = 48, 18
    hcf = calculate_hcf(num1, num2)
    lcm = calculate_lcm(num1, num2)
    print(f"HCF of {num1} and {num2} is: {hcf}")
    print(f"LCM of {num1} and {num2} is: {lcm}")

    # --- Physics and Geometry ---
    print("\n--- Physics and Geometry ---")
    
    # Area of Circle
    radius = 5
    area = area_circle(radius)
    print(f"The area of a circle with radius {radius} is: {area:.2f}")

    # Kinetic Energy
    mass, velocity = 10, 5 # kg, m/s
    ke = calculate_kinetic_energy(mass, velocity)
    print(f"Kinetic energy of a {mass}kg object at {velocity}m/s is: {ke} Joules")

    # Demonstrate the number guessing game
    # Uncomment the line below to play the game interactively.
    # number_guessing_game()

if __name__ == "__main__":
    main()
