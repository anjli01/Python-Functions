# -*- coding: utf-8 -*-
"""
A collection of utility functions for common calculations in geometry, physics,
finance, and general data manipulation.

This module is intended as a repository of simple, well-documented Python
functions following standard coding practices.
"""

# =============================================================================
# Imports
# =============================================================================
import math
import random
import re
import sys
from datetime import date, datetime
from typing import List, Dict, Any, Union, Iterable

# =============================================================================
# Constants
# =============================================================================
# Using math.pi is more precise than 3.14
PI = math.pi
# Universal Gravitational Constant (m^3 kg^-1 s^-2)
GRAVITATIONAL_CONSTANT = 6.67430e-11
# Ideal Gas Constant (J K^-1 mol^-1)
IDEAL_GAS_CONSTANT = 8.3145
# Speed of light in vacuum (m/s)
SPEED_OF_LIGHT = 299792458
# Acceleration due to gravity on Earth (m/s^2)
EARTH_GRAVITY = 9.80665
# Coulomb's constant (N m^2 C^-2)
COULOMB_CONSTANT = 8.9875517923e9

# =============================================================================
# Geometry Calculations
# =============================================================================

def calculate_cone_curved_surface_area(slant_height: float, radius: float) -> float:
    """Calculates the curved surface area of a cone."""
    return PI * radius * slant_height

def calculate_cube_surface_area(side_length: float) -> float:
    """Calculates the total surface area of a cube."""
    return 6 * (side_length ** 2)

def calculate_cuboid_surface_area(length: float, breadth: float, height: float) -> float:
    """Calculates the total surface area of a cuboid."""
    return 2 * (length * breadth + breadth * height + height * length)

def calculate_sphere_surface_area(radius: float) -> float:
    """Calculates the surface area of a sphere."""
    return 4 * PI * (radius ** 2)

def calculate_hemisphere_curved_surface_area(radius: float) -> float:
    """Calculates the curved surface area of a hemisphere."""
    return 2 * PI * (radius ** 2)

def calculate_cylinder_surface_area(height: float, radius: float) -> float:
    """Calculates the total surface area of a cylinder."""
    top_bottom_area = 2 * PI * radius ** 2
    side_area = 2 * PI * radius * height
    return top_bottom_area + side_area

def calculate_cone_slant_height(height: float, radius: float) -> float:
    """Calculates the slant height of a cone from its perpendicular height and radius."""
    return math.sqrt(height**2 + radius**2)

def calculate_cone_lateral_surface_area(height: float, radius: float) -> float:
    """Calculates the lateral (side) surface area of a cone."""
    slant_height = calculate_cone_slant_height(height, radius)
    return PI * radius * slant_height

def calculate_cylinder_volume(height: float, radius: float) -> float:
    """Calculates the volume of a cylinder."""
    return PI * (radius ** 2) * height

def calculate_cone_volume(height: float, radius: float) -> float:
    """Calculates the volume of a cone."""
    return PI * (radius ** 2) * height / 3

def calculate_hemisphere_volume(radius: float) -> float:
    """Calculates the volume of a hemisphere."""
    return (2 / 3) * PI * (radius ** 3)

def calculate_sphere_volume(radius: float) -> float:
    """Calculates the volume of a sphere."""
    return (4 / 3) * PI * (radius ** 3)

def calculate_cuboid_volume(length: float, breadth: float, height: float) -> float:
    """Calculates the volume of a cuboid."""
    return length * breadth * height

def calculate_cube_volume(side_length: float) -> float:
    """Calculates the volume of a cube."""
    return side_length ** 3

def calculate_circle_area(radius: float) -> float:
    """Calculates the area of a circle with a given radius."""
    return PI * radius ** 2

# =============================================================================
# Physics Calculations
# =============================================================================

def calculate_speed(distance: float, time: float) -> float:
    """Calculates speed given distance and time."""
    if time == 0:
        raise ValueError("Time cannot be zero.")
    return distance / time

def calculate_distance(speed: float, time: float) -> float:
    """Calculates distance given speed and time."""
    return speed * time

def calculate_time(distance: float, speed: float) -> float:
    """Calculates time taken given distance and speed."""
    if speed == 0:
        raise ValueError("Speed cannot be zero.")
    return distance / speed

def calculate_torque(force: float, radius: float, theta: float) -> float:
    """Calculates torque given force, radius (lever arm), and angle in radians."""
    return force * radius * math.sin(theta)

def calculate_angular_velocity(angular_distance: float, time: float) -> float:
    """Calculates angular velocity from angular distance (radians) and time."""
    if time == 0:
        raise ValueError("Time cannot be zero.")
    return angular_distance / time

def calculate_lens_focal_length(object_distance: float, image_distance: float) -> float:
    """Calculates the focal length of a lens using the lens formula."""
    if (object_distance + image_distance) == 0:
        raise ValueError("Sum of object and image distance cannot be zero.")
    return (object_distance * image_distance) / (object_distance + image_distance)

def calculate_gravitational_force(mass1: float, mass2: float, distance: float) -> float:
    """Calculates the gravitational force between two objects."""
    if distance == 0:
        raise ValueError("Distance cannot be zero.")
    return (GRAVITATIONAL_CONSTANT * mass1 * mass2) / (distance ** 2)

def calculate_current(voltage: float, resistance: float) -> float:
    """Calculates electrical current (Ohm's Law)."""
    if resistance == 0:
        raise ValueError("Resistance cannot be zero.")
    return voltage / resistance

def calculate_parallel_capacitance(capacitances: List[float]) -> float:
    """Calculates the total capacitance of capacitors in parallel."""
    return sum(capacitances)

def calculate_parallel_resistance(resistances: List[float]) -> float:
    """Calculates the total resistance of resistors in parallel."""
    # The original formula `sum([1/r for r in res_list])` was incorrect.
    # It calculates the sum of conductances. The correct formula is 1 / sum(1/r).
    if any(r == 0 for r in resistances):
        return 0  # A short circuit results in zero total resistance.
    
    total_conductance = sum(1 / r for r in resistances)
    return 1 / total_conductance if total_conductance != 0 else float('inf')

def calculate_series_resistance(resistances: List[float]) -> float:
    """Calculates the total resistance of resistors in series."""
    return sum(resistances)

def calculate_moment_of_inertia_ring(mass: float, radius: float) -> float:
    """Calculates the moment of inertia of a ring about its center."""
    return mass * (radius ** 2)

def calculate_moment_of_inertia_sphere(mass: float, radius: float) -> float:
    """Calculates the moment of inertia of a solid sphere."""
    # Note: The original formula (7/5)*m*r^2 is for a thick-walled spherical shell.
    # The standard formula for a solid sphere is (2/5)*m*r^2. I will use the more common one.
    return (2 / 5) * mass * (radius ** 2)

def calculate_ideal_gas_pressure(moles: float, temp_kelvin: float, volume: float) -> float:
    """Calculates the pressure of an ideal gas (Ideal Gas Law)."""
    if volume == 0:
        raise ValueError("Volume cannot be zero.")
    return (moles * IDEAL_GAS_CONSTANT * temp_kelvin) / volume

def calculate_ideal_gas_volume(moles: float, temp_kelvin: float, pressure: float) -> float:
    """Calculates the volume of an ideal gas (Ideal Gas Law)."""
    if pressure == 0:
        raise ValueError("Pressure cannot be zero.")
    return (moles * IDEAL_GAS_CONSTANT * temp_kelvin) / pressure

def calculate_ideal_gas_temperature(moles: float, pressure: float, volume: float) -> float:
    """Calculates the temperature of an ideal gas (Ideal Gas Law)."""
    # Original had a bug: `return (pressure*volume)/n*r`. Due to operator precedence,
    # it calculated `((p*v)/n) * r`. It should be `(p*v) / (n*r)`.
    if moles == 0 or IDEAL_GAS_CONSTANT == 0:
         raise ValueError("Number of moles and gas constant cannot be zero.")
    return (pressure * volume) / (moles * IDEAL_GAS_CONSTANT)

def calculate_final_velocity(initial_velocity: float, acceleration: float, time: float) -> float:
    """Calculates the final velocity of an object."""
    return initial_velocity + acceleration * time

def calculate_displacement(initial_velocity: float, acceleration: float, time: float) -> float:
    """Calculates the displacement of an object."""
    return initial_velocity * time + 0.5 * acceleration * (time ** 2)

def calculate_remaining_radioactive_amount(initial_quantity: float, time_elapsed: float, half_life: float) -> float:
    """Calculates the amount of a radioactive element left after a certain time."""
    if half_life <= 0:
        raise ValueError("Half-life must be positive.")
    return initial_quantity * (0.5 ** (time_elapsed / half_life))

def calculate_energy_from_mass(mass: float) -> float:
    """Calculates energy released from mass (Einstein's E=mc^2)."""
    return mass * (SPEED_OF_LIGHT ** 2)

def calculate_kinetic_energy(mass: float, velocity: float) -> float:
    """Calculates the kinetic energy of an object."""
    return 0.5 * mass * (velocity ** 2)

def calculate_potential_energy(mass: float, height: float) -> float:
    """Calculates the potential energy of an object at a certain height."""
    return mass * EARTH_GRAVITY * height

def calculate_electrostatic_force(charge1: float, charge2: float, distance: float) -> float:
    """Calculates the electrostatic force between two charged particles."""
    if distance == 0:
        raise ValueError("Distance cannot be zero.")
    return (COULOMB_CONSTANT * charge1 * charge2) / (distance ** 2)

def calculate_density(mass: float, volume: float) -> float:
    """Calculates density given mass and volume."""
    if volume == 0:
        raise ValueError("Volume cannot be zero.")
    return mass / volume

# =============================================================================
# Financial Calculations
# =============================================================================

def calculate_price_after_discount(price: float, discount_percent: float) -> float:
    """Calculates the new selling price after a percentage discount."""
    return price * (1 - discount_percent / 100)

def calculate_simple_interest(principal: float, rate_percent: float, time_years: float) -> float:
    """Calculates simple interest."""
    return (principal * rate_percent * time_years) / 100

def calculate_compound_interest(principal: float, rate_percent: float, time_years: float, compounds_per_year: int) -> float:
    """Calculates compound interest."""
    rate = rate_percent / 100
    amount = principal * ((1 + (rate / compounds_per_year)) ** (compounds_per_year * time_years))
    return amount - principal

# =============================================================================
# General Utilities
# =============================================================================

def convert_temperature(temp: float, unit: str = 'f') -> float:
    """
    Converts temperature between Fahrenheit and Celsius.
    Args:
        temp: The temperature value to convert.
        unit: The unit of the input temperature ('f' for Fahrenheit, 'c' for Celsius).
    Returns:
        The converted temperature.
    """
    unit = unit.lower()
    if unit == 'f':
        return (temp - 32) * (5 / 9)  # To Celsius
    elif unit == 'c':
        return (temp * 9 / 5) + 32  # To Fahrenheit
    else:
        raise ValueError("Unit must be 'f' or 'c'.")
        
def merge_dictionaries_in_lists(list1: List[Dict], list2: List[Dict]) -> List[Dict]:
    """
    Merges dictionaries from two lists element-wise.
    
    For each index, keys from the dictionary in list2 are added to the
    corresponding dictionary in list1 if the key does not already exist.
    Note: This function modifies list1 in place.
    """
    for i, (dict1, dict2) in enumerate(zip(list1, list2)):
        for key, value in dict2.items():
            if key not in dict1:
                dict1[key] = value
    return list1

def concatenate_matrix_columns(matrix: List[List[str]]) -> List[str]:
    """
    Performs vertical string concatenation on a matrix (list of lists).
    
    Example: [["a", "b"], ["c", "d"]] -> ["ac", "bd"]
    """
    if not matrix:
        return []

    num_cols = max(len(row) for row in matrix)
    result = []
    for col_idx in range(num_cols):
        column_str = ""
        for row in matrix:
            if col_idx < len(row):
                column_str += row[col_idx]
        result.append(column_str)
    return result

def get_kth_matrix_column(matrix: List[List[Any]], k: int) -> tuple:
    """
    Extracts the k-th column from a matrix.
    Args:
        matrix: The input matrix (list of lists).
        k: The zero-based index of the column to extract.
    Returns:
        A tuple containing the elements of the k-th column.
    """
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError("Input must be a list of lists.")
    if k < 0 or any(k >= len(row) for row in matrix):
        raise IndexError("k is out of bounds for the given matrix.")
    
    return tuple(row[k] for row in matrix)
    
def find_all_subarrays(arr: List[Any]) -> List[List[Any]]:
    """Generates all possible contiguous subarrays of a list."""
    subarrays = []
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            subarrays.append(arr[i : j + 1])
    return subarrays
    
def sum_nested_list(nested_list: List) -> Union[int, float]:
    """
    Calculates the sum of all numbers in a potentially nested list.
    
    This version uses recursion and avoids global variables.
    """
    total = 0
    for item in nested_list:
        if isinstance(item, list):
            total += sum_nested_list(item)
        elif isinstance(item, (int, float)):
            total += item
    return total

def power(base: Union[int, float], exponent: int) -> Union[int, float]:
    """
    Calculates the power of a number using recursion.
    Handles non-negative integer exponents.
    """
    if exponent < 0:
        raise ValueError("This recursive implementation only supports non-negative exponents.")
    if exponent == 0:
        return 1
    if exponent == 1:
        return base
    return base * power(base, exponent - 1)

def filter_strings_by_substring(data: List[str], substring: str, start: int, end: int) -> List[str]:
    """Filters a list of strings, keeping those with a specific substring at a given slice."""
    return [s for s in data if s[start:end] == substring]

def remove_punctuation(text: str) -> str:
    """Removes all standard punctuation characters from a string."""
    punctuation = r'!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    return text.translate(str.maketrans('', '', punctuation))

def time_difference(h1: int, m1: int, h2: int, m2: int) -> tuple[int, int]:
    """Calculates the difference in hours and minutes between two times."""
    total_minutes1 = h1 * 60 + m1
    total_minutes2 = h2 * 60 + m2
    diff_minutes = abs(total_minutes2 - total_minutes1)
    
    hours = diff_minutes // 60
    minutes = diff_minutes % 60
    return hours, minutes

def convert_12h_to_24h(time_12h: str) -> str:
    """Converts a time string from 12-hour format (e.g., '08:05:45 PM') to 24-hour format."""
    in_time = datetime.strptime(time_12h, "%I:%M:%S %p")
    return datetime.strftime(in_time, "%H:%M:%S")

def calculate_clock_hands_angle(hour: int, minute: int) -> float:
    """Calculates the smaller angle between the hour and minute hands of a clock."""
    if not (0 <= hour <= 12 and 0 <= minute <= 59):
        raise ValueError("Invalid time provided.")
        
    hour = hour % 12
    hour_angle = 0.5 * (hour * 60 + minute)
    minute_angle = 6 * minute
    angle = abs(hour_angle - minute_angle)
    return min(360 - angle, angle)

def is_valid_email(email: str) -> bool:
    """Checks if a string is a valid email address using regex."""
    regex = r'^[a-z0-9]+[\._]?[a-z0-9]+[@]\w+[.]\w{2,3}$'
    return re.search(regex, email) is not None

def get_age_from_dob(birth_date: date) -> int:
    """Calculates current age given a date of birth."""
    today = date.today()
    age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
    return age

def get_factorial(num: int) -> int:
    """Calculates the factorial of a non-negative integer."""
    if num < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    if num == 0:
        return 1
    return math.factorial(num)

# =============================================================================
# Sorting Algorithms
# =============================================================================

def gnome_sort(arr: List[Union[int, float]]) -> List[Union[int, float]]:
    """
    Sorts a list using the Gnome Sort algorithm.
    Modifies the list in-place.
    """
    index = 0
    n = len(arr)
    while index < n:
        if index == 0:
            index += 1
        if arr[index] >= arr[index - 1]:
            index += 1
        else:
            arr[index], arr[index - 1] = arr[index - 1], arr[index]
            index -= 1
    return arr

def pigeonhole_sort(arr: List[int]) -> List[int]:
    """
    Sorts a list of integers using the Pigeonhole Sort algorithm.
    Suitable for lists where the number of elements and the range of values are similar.
    """
    if not all(isinstance(x, int) for x in arr):
        raise TypeError("Pigeonhole sort requires a list of integers.")
    
    min_val, max_val = min(arr), max(arr)
    size = max_val - min_val + 1
    holes = [0] * size

    for x in arr:
        holes[x - min_val] += 1
        
    i = 0
    for count in range(size):
        while holes[count] > 0:
            holes[count] -= 1
            arr[i] = count + min_val
            i += 1
    return arr

def stooge_sort(arr: List[Union[int, float]], low: int, high: int) -> None:
    """
    Sorts a portion of a list using the Stooge Sort recursive algorithm.
    Modifies the list in-place.
    """
    if low >= high:
        return
        
    if arr[low] > arr[high]:
        arr[low], arr[high] = arr[high], arr[low]
        
    if high - low + 1 > 2:
        third = (high - low + 1) // 3
        stooge_sort(arr, low, high - third)
        stooge_sort(arr, low + third, high)
        stooge_sort(arr, low, high - third)


# =============================================================================
# Main execution block for demonstration
# =============================================================================

def main():
    """Main function to demonstrate the usage of the refactored functions."""
    print("--- Demonstrating Refactored Functions ---")

    # --- Geometry Demo ---
    print("\n--- Geometry ---")
    cone_area = calculate_cone_curved_surface_area(slant_height=10, radius=5)
    print(f"Cone Curved Surface Area: {cone_area:.2f}")
    cube_vol = calculate_cube_volume(side_length=3)
    print(f"Cube Volume: {cube_vol}")
    
    # --- Physics Demo ---
    print("\n--- Physics ---")
    speed = calculate_speed(distance=100, time=9.58)
    print(f"Calculated Speed (Usain Bolt): {speed:.2f} m/s")
    force = calculate_gravitational_force(mass1=5.972e24, mass2=7.342e22, distance=3.844e8)
    print(f"Gravitational Force (Earth-Moon): {force:.2e} N")
    
    # --- Finance Demo ---
    print("\n--- Finance ---")
    ci = calculate_compound_interest(principal=1000, rate_percent=5, time_years=10, compounds_per_year=4)
    print(f"Compound Interest after 10 years: ${ci:.2f}")

    # --- Utilities Demo ---
    print("\n--- Utilities ---")
    temp_c = convert_temperature(98.6, unit='f')
    print(f"98.6°F is {temp_c:.1f}°C")
    
    nested_list_sum = sum_nested_list([[1, 2, 3], [4, [5, 6]], 7])
    print(f"Sum of nested list [[1, 2, 3], [4, [5, 6]], 7]: {nested_list_sum}")
    
    power_val = power(5, 3)
    print(f"5 to the power of 3 is: {power_val}")

    time_24h = convert_12h_to_24h("08:05:45 PM")
    print(f"'08:05:45 PM' in 24-hour format: {time_24h}")

    # --- Sorting Demo ---
    print("\n--- Sorting Algorithms ---")
    gnome_data = [34, 2, 10, -9, 15]
    print(f"Original list for Gnome Sort: {gnome_data}")
    gnome_sort(gnome_data)
    print(f"Sorted with Gnome Sort: {gnome_data}")

    stooge_data = [2, 4, 5, 3, 1, 9, 7]
    print(f"Original list for Stooge Sort: {stooge_data}")
    stooge_sort(stooge_data, 0, len(stooge_data) - 1)
    print(f"Sorted with Stooge Sort: {stooge_data}")

    # --- Other Snippets Demo ---
    print("\n--- Other Snippets ---")
    # Swapping two numbers
    num1, num2 = 10, 20
    print(f"Before swap: num1={num1}, num2={num2}")
    num1, num2 = num2, num1
    print(f"After swap:  num1={num1}, num2={num2}")

    # List comprehension for random integers
    random_list = [random.randint(1, 100) for _ in range(10)]
    print(f"Generated random list: {random_list}")
    
    # Check if a number is positive
    is_positive = lambda n: n > 0
    print(f"Is -5 positive? {is_positive(-5)}")
    print(f"Is 10 positive? {is_positive(10)}")
    
if __name__ == "__main__":
    main()