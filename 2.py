# -*- coding: utf-8 -*-
"""
A comprehensive collection of utility functions for mathematical, scientific,
and general-purpose programming tasks in Python.

This module is organized into the following sections:
- Constants: Mathematical and physical constants.
- Geometry: Functions for calculating area, perimeter, volume, etc.
- Algebra: Functions for solving quadratic equations.
- Physics: Formulas from kinematics, dynamics, electricity, and more.
- Finance: Functions for calculating interest, profit, and loss.
- Data Structures & Algorithms: Sorting, list/dict manipulation, etc.
- Decorators & Higher-Order Functions: Utilities for function modification.
- Time Utilities: Functions for time conversion and calculations.
- String Utilities: Functions for string manipulation.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
import math
import operator
import string
import itertools
import time
from collections import deque
from functools import wraps, reduce
from heapq import heappush, heappop
from typing import (
    List, Tuple, Optional, Union, Any, Callable, Iterator, TypeVar, Dict, Set
)

# For the profanity filter example, these would be required.
# from urllib.request import urlopen
# from bs4 import BeautifulSoup


# =============================================================================
# 2. CONSTANTS
# =============================================================================
# Using constants improves readability, maintainability, and precision.

# Mathematical Constants
PI = math.pi

# Physical Constants (SI units)
SPEED_OF_LIGHT: float = 299792458.0  # m/s
STANDARD_GRAVITY: float = 9.80665  # m/s^2
GRAVITATIONAL_CONSTANT: float = 6.67430e-11  # m^3 kg^-1 s^-2
IDEAL_GAS_CONSTANT: float = 8.314462618  # J/(mol·K)
COULOMB_CONSTANT: float = 8.9875517923e9  # N·m^2/C^2


# =============================================================================
# 3. GEOMETRY FUNCTIONS
# =============================================================================

def calculate_cartesian_distance(
    p1: Tuple[float, float], p2: Tuple[float, float]
) -> float:
    """Calculates the Cartesian distance between two points in a 2D plane.

    Args:
        p1: A tuple (x1, y1) representing the first point.
        p2: A tuple (x2, y2) representing the second point.

    Returns:
        The Euclidean distance between the two points.
    """
    # math.hypot is more accurate and efficient for this calculation.
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def calculate_area_of_rectangle(length: float, breadth: float) -> float:
    """Calculates the area of a rectangle."""
    return length * breadth


def calculate_area_of_square(side: float) -> float:
    """Calculates the area of a square."""
    return side ** 2


def calculate_area_of_rhombus(d1: float, d2: float) -> float:
    """Calculates the area of a rhombus given its two diagonals."""
    return (d1 * d2) / 2


def calculate_area_of_trapezium(base1: float, base2: float, height: float) -> float:
    """Calculates the area of a trapezium."""
    return height * (base1 + base2) / 2


def calculate_area_of_circle(radius: float) -> float:
    """Calculates the area of a circle."""
    return PI * radius ** 2


def calculate_circumference_of_circle(radius: float) -> float:
    """Calculates the circumference of a circle."""
    return 2 * PI * radius


def calculate_perimeter_of_rectangle(length: float, breadth: float) -> float:
    """Calculates the perimeter of a rectangle."""
    return 2 * (length + breadth)


def calculate_perimeter_of_triangle(s1: float, s2: float, s3: float) -> float:
    """Calculates the perimeter of a triangle."""
    return s1 + s2 + s3


def calculate_perimeter_of_square(side: float) -> float:
    """Calculates the perimeter of a square."""
    return 4 * side


def calculate_perimeter_of_equilateral_triangle(side: float) -> float:
    """Calculates the perimeter of an equilateral triangle."""
    return 3 * side


def calculate_perimeter_of_isosceles_triangle(equal_side: float, base: float) -> float:
    """Calculates the perimeter of an isosceles triangle."""
    return 2 * equal_side + base


def calculate_area_of_ellipse(minor_axis: float, major_axis: float) -> float:
    """Calculates the area of an ellipse."""
    return PI * minor_axis * major_axis


# --- 3D Shapes ---

def calculate_lateral_surface_area_of_cylinder(radius: float, height: float) -> float:
    """Calculates the lateral surface area of a cylinder."""
    return 2 * PI * radius * height


def calculate_curved_surface_area_of_cone(radius: float, slant_height: float) -> float:
    """Calculates the curved surface area of a cone."""
    return PI * radius * slant_height


def calculate_total_surface_area_of_cube(side: float) -> float:
    """Calculates the total surface area of a cube."""
    return 6 * (side ** 2)


def calculate_total_surface_area_of_cuboid(
    length: float, breadth: float, height: float
) -> float:
    """Calculates the total surface area of a cuboid."""
    return 2 * (length * breadth + breadth * height + height * length)


def calculate_surface_area_of_sphere(radius: float) -> float:
    """Calculates the surface area of a sphere."""
    return 4 * PI * (radius ** 2)


def calculate_curved_surface_area_of_hemisphere(radius: float) -> float:
    """Calculates the curved surface area of a hemisphere."""
    return 2 * PI * (radius ** 2)


def calculate_total_surface_area_of_cylinder(radius: float, height: float) -> float:
    """Calculates the total surface area of a cylinder."""
    # FIX: Corrected typo from `*+` to `+`.
    return 2 * PI * radius * (radius + height)


def calculate_lateral_surface_area_of_cone(radius: float, height: float) -> float:
    """Calculates the lateral surface area of a cone using radius and height."""
    slant_height = math.hypot(radius, height)
    return PI * radius * slant_height


def calculate_volume_of_cylinder(radius: float, height: float) -> float:
    """Calculates the volume of a cylinder."""
    return PI * (radius ** 2) * height


def calculate_volume_of_cone(radius: float, height: float) -> float:
    """Calculates the volume of a cone."""
    return PI * (radius ** 2) * height / 3


def calculate_volume_of_hemisphere(radius: float) -> float:
    """Calculates the volume of a hemisphere."""
    return (2 / 3) * PI * (radius ** 3)


def calculate_volume_of_sphere(radius: float) -> float:
    """Calculates the volume of a sphere."""
    return (4 / 3) * PI * (radius ** 3)


def calculate_volume_of_cuboid(length: float, breadth: float, height: float) -> float:
    """Calculates the volume of a cuboid."""
    return length * breadth * height


def calculate_volume_of_cube(side: float) -> float:
    """Calculates the volume of a cube."""
    return side ** 3


# =============================================================================
# 4. ALGEBRA FUNCTIONS (Quadratic Equations: ax^2 + bx + c = 0)
# =============================================================================

def _calculate_discriminant(a: float, b: float, c: float) -> float:
    """Helper function to calculate the discriminant of a quadratic equation."""
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero for a quadratic equation.")
    return b ** 2 - 4 * a * c


def get_quadratic_root_type(a: float, b: float, c: float) -> str:
    """Determines the type of roots of a quadratic equation.

    Returns:
        'real and distinct', 'real and equal', or 'complex'.
    """
    discriminant = _calculate_discriminant(a, b, c)
    if discriminant > 0:
        return 'real and distinct'
    if discriminant == 0:
        return 'real and equal'
    return 'complex'


def get_sum_of_quadratic_roots(a: float, b: float, _c: float) -> float:
    """Calculates the sum of the roots of a quadratic equation.

    Formula: -b / a
    """
    # FIX: The original function was incorrect. It returned c/a.
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero.")
    return -b / a


def get_product_of_quadratic_roots(a: float, _b: float, c: float) -> float:
    """Calculates the product of the roots of a quadratic equation.

    Formula: c / a
    """
    # FIX: The original function was incorrect. It returned -b/a.
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero.")
    return c / a


def calculate_real_quadratic_roots(
    a: float, b: float, c: float
) -> Optional[Tuple[float, float]]:
    """Calculates the real roots of a quadratic equation.

    Returns:
        A tuple containing the two real roots, or None if the roots are complex.
    """
    discriminant = _calculate_discriminant(a, b, c)
    if discriminant < 0:
        return None

    # FIX: Corrected parenthesis error. Division by 2*a must be grouped.
    sqrt_d = math.sqrt(discriminant)
    root1 = (-b + sqrt_d) / (2 * a)
    root2 = (-b - sqrt_d) / (2 * a)
    return root1, root2


# =============================================================================
# 5. FINANCE FUNCTIONS
# =============================================================================

def calculate_profit_or_loss(cost_price: float, selling_price: float) -> Tuple[str, float]:
    """Calculates the profit, loss, or break-even status and amount."""
    if selling_price > cost_price:
        return 'profit', selling_price - cost_price
    if selling_price < cost_price:
        return 'loss', cost_price - selling_price
    return 'no profit or loss', 0.0


def calculate_selling_price_after_discount(
    original_price: float, discount_percent: float
) -> float:
    """Calculates the final price after applying a discount."""
    if not 0 <= discount_percent <= 100:
        raise ValueError("Discount percentage must be between 0 and 100.")
    return original_price * (1 - discount_percent / 100)


def calculate_simple_interest(
    principal: float, rate_percent: float, time_years: float
) -> float:
    """Calculates simple interest."""
    return (principal * rate_percent * time_years) / 100


def calculate_compound_interest(
    principal: float,
    rate_percent: float,
    time_years: float,
    compounds_per_year: int,
) -> float:
    """Calculates compound interest."""
    amount = principal * (
        (1 + (rate_percent / (compounds_per_year * 100)))
        ** (compounds_per_year * time_years)
    )
    return amount - principal


# =============================================================================
# 6. PHYSICS FUNCTIONS
# =============================================================================

# --- Kinematics ---

def calculate_speed(distance: float, time: float) -> float:
    """Calculates speed given distance and time."""
    if time == 0:
        return math.inf if distance > 0 else 0.0
    return distance / time


def calculate_distance(speed: float, time: float) -> float:
    """Calculates distance given speed and time."""
    return speed * time


def calculate_time(distance: float, speed: float) -> float:
    """Calculates time given distance and speed."""
    if speed == 0:
        return math.inf if distance > 0 else 0.0
    return distance / speed


def calculate_final_velocity(
    initial_velocity: float, acceleration: float, time: float
) -> float:
    """Calculates final velocity using v = u + at."""
    return initial_velocity + acceleration * time


def calculate_displacement(
    initial_velocity: float, acceleration: float, time: float
) -> float:
    """Calculates displacement using s = ut + 0.5at^2."""
    return initial_velocity * time + 0.5 * acceleration * (time ** 2)


# --- Dynamics & Energy ---

def calculate_torque(force: float, radius: float, angle_rad: float) -> float:
    """Calculates torque given force, lever arm radius, and angle in radians."""
    return force * radius * math.sin(angle_rad)


def calculate_angular_velocity(angular_distance_rad: float, time: float) -> float:
    """Calculates angular velocity."""
    if time == 0:
        return math.inf if angular_distance_rad > 0 else 0.0
    return angular_distance_rad / time


def calculate_gravitational_force(m1: float, m2: float, distance: float) -> float:
    """Calculates the gravitational force between two masses."""
    if distance == 0:
        raise ValueError("Distance cannot be zero.")
    return (GRAVITATIONAL_CONSTANT * m1 * m2) / (distance ** 2)


def calculate_energy_from_mass(mass: float) -> float:
    """Calculates energy released from mass conversion (E=mc^2)."""
    # FIX: Used precise speed of light constant.
    return mass * (SPEED_OF_LIGHT ** 2)


def calculate_kinetic_energy(mass: float, velocity: float) -> float:
    """Calculates kinetic energy."""
    return 0.5 * mass * (velocity ** 2)


def calculate_potential_energy(mass: float, height: float) -> float:
    """Calculates gravitational potential energy near Earth's surface."""
    return mass * STANDARD_GRAVITY * height


def calculate_density(mass: float, volume: float) -> float:
    """Calculates density."""
    if volume == 0:
        raise ValueError("Volume cannot be zero.")
    return mass / volume


# --- Electricity & Magnetism ---

def calculate_focal_length_of_lens(obj_dist: float, img_dist: float) -> float:
    """Calculates the focal length of a lens using the lens formula."""
    if (obj_dist + img_dist) == 0:
        raise ValueError("Sum of object and image distance cannot be zero.")
    return (obj_dist * img_dist) / (obj_dist + img_dist)


def calculate_current(voltage: float, resistance: float) -> float:
    """Calculates current using Ohm's law (I = V/R)."""
    if resistance == 0:
        return math.inf if voltage != 0 else 0.0  # Ideal conductor
    return voltage / resistance


def calculate_parallel_capacitance(capacitances: List[float]) -> float:
    """Calculates the total capacitance of capacitors in parallel."""
    return sum(capacitances)


def calculate_parallel_resistance(resistances: List[float]) -> float:
    """Calculates the total resistance of resistors in parallel."""
    # FIX: Corrected formula. The original returned the sum of reciprocals.
    # Also added a check for zero resistance.
    if any(r == 0 for r in resistances):
        return 0.0  # Short circuit
    if not resistances:
        return math.inf  # Open circuit
    total_conductance = sum(1 / r for r in resistances)
    return 1 / total_conductance


def calculate_series_resistance(resistances: List[float]) -> float:
    """Calculates the total resistance of resistors in series."""
    return sum(resistances)


def calculate_electrostatic_force(q1: float, q2: float, distance: float) -> float:
    """Calculates the electrostatic force between two point charges."""
    if distance == 0:
        raise ValueError("Distance cannot be zero.")
    return (COULOMB_CONSTANT * q1 * q2) / (distance ** 2)


# --- Moment of Inertia ---

def calculate_mi_of_ring(mass: float, radius: float) -> float:
    """Calculates the moment of inertia of a ring about its center."""
    return mass * (radius ** 2)


def calculate_mi_of_solid_sphere(mass: float, radius: float) -> float:
    """Calculates the moment of inertia of a solid sphere about its center."""
    # FIX: Corrected formula. The common formula is (2/5)mr^2 for a solid sphere.
    return (2 / 5) * mass * (radius ** 2)


# --- Thermodynamics ---

def find_pressure_of_ideal_gas(n_moles: float, temp_k: float, volume: float) -> float:
    """Calculates the pressure of an ideal gas (PV=nRT)."""
    if volume <= 0:
        raise ValueError("Volume must be positive.")
    return (n_moles * IDEAL_GAS_CONSTANT * temp_k) / volume


def find_volume_of_ideal_gas(n_moles: float, temp_k: float, pressure: float) -> float:
    """Calculates the volume of an ideal gas (PV=nRT)."""
    if pressure <= 0:
        raise ValueError("Pressure must be positive.")
    return (n_moles * IDEAL_GAS_CONSTANT * temp_k) / pressure


def find_temp_of_ideal_gas(n_moles: float, pressure: float, volume: float) -> float:
    """Calculates the temperature of an ideal gas (PV=nRT)."""
    # FIX: Corrected operator precedence. Division must be grouped.
    if n_moles <= 0:
        raise ValueError("Number of moles must be positive.")
    return (pressure * volume) / (n_moles * IDEAL_GAS_CONSTANT)


# --- Nuclear Physics ---

def calculate_remaining_radioactive_mass(
    initial_quantity: float, time_elapsed: float, half_life: float
) -> float:
    """Calculates the amount of a radioactive element left after a certain time."""
    if half_life <= 0:
        raise ValueError("Half-life must be positive.")
    return initial_quantity * (0.5 ** (time_elapsed / half_life))


# =============================================================================
# 7. MISCELLANEOUS UTILITIES
# =============================================================================

# --- Temperature Conversion ---

def convert_temperature(temp: float, unit_from: str = 'f') -> float:
    """Converts temperature between Fahrenheit and Celsius.

    Args:
        temp: The temperature value to convert.
        unit_from: The unit of the input temperature ('f' or 'c').

    Returns:
        The converted temperature.
    """
    unit = unit_from.lower()
    if unit == 'f' or unit == 'fahrenheit':
        return (temp - 32) * (5 / 9)  # To Celsius
    if unit == 'c' or unit == 'celsius':
        return (temp * 9 / 5) + 32    # To Fahrenheit
    raise ValueError("Invalid unit. Must be 'f' or 'c'.")


# --- Data Structure Manipulation ---

def merge_list_of_dicts(
    list1: List[Dict], list2: List[Dict]
) -> List[Dict]:
    """Merges two lists of dictionaries element-wise, updating dicts in list1.

    Note: Assumes lists are of the same length. Creates a new list to avoid
          modifying the original list in place.
    """
    if len(list1) != len(list2):
        raise ValueError("Input lists must have the same length.")

    merged_list = [d1.copy() for d1 in list1]  # Avoid modifying original
    for i, d2 in enumerate(list2):
        merged_list[i].update(d2)
    return merged_list


def concatenate_matrix_columns(matrix: List[List[str]]) -> List[str]:
    """Concatenates the columns of a matrix of strings.

    Example: [["a", "b"], ["c", "d"]] -> ["ac", "bd"]
    """
    # Using zip_longest handles uneven column lengths gracefully.
    zipped_cols = itertools.zip_longest(*matrix, fillvalue='')
    return ["".join(col) for col in zipped_cols]


def get_kth_matrix_column(matrix: List[List[Any]], k: int) -> List[Any]:
    """Extracts the Kth column from a matrix (list of lists)."""
    if not matrix or not all(isinstance(row, list) for row in matrix):
        raise TypeError("Input must be a list of lists.")
    if k < 0:
        raise IndexError("Column index cannot be negative.")
        
    column = []
    for row in matrix:
        if k < len(row):
            column.append(row[k])
        else:
            # Handle rows that are shorter than the requested column index
            column.append(None)
    return column


def generate_all_subarrays(arr: List[Any]) -> Iterator[List[Any]]:
    """A generator that yields all contiguous subarrays of a list."""
    n = len(arr)
    for i in range(n):
        for j in range(i, n):
            yield arr[i : j + 1]


def sum_nested_list_recursive(nested_list: List) -> Union[int, float]:
    """Calculates the sum of all numbers in a potentially nested list."""
    # FIX: Removed the use of a global variable for a pure functional approach.
    total = 0
    for item in nested_list:
        if isinstance(item, list):
            total += sum_nested_list_recursive(item)
        elif isinstance(item, (int, float)):
            total += item
    return total


def power_recursive(base: Union[int, float], exp: int) -> Union[int, float]:
    """Calculates power using recursion."""
    if exp < 0:
        return 1 / power_recursive(base, -exp)
    if exp == 0:
        return 1
    if exp == 1:
        return base
    return base * power_recursive(base, exp - 1)


# --- String Manipulation ---

def filter_strings_by_substring_at_pos(
    data: List[str], substring: str, start: int
) -> List[str]:
    """Filters a list of strings for those containing a substring at a specific position."""
    end = start + len(substring)
    return [s for s in data if s[start:end] == substring]


def remove_punctuation(text: str) -> str:
    """Removes all punctuation from a string efficiently."""
    # Using str.translate is much more efficient than looping and replacing.
    return text.translate(str.maketrans('', '', string.punctuation))


# =============================================================================
# 8. ALGORITHMS
# =============================================================================

# --- Sorting Algorithms ---

def gnome_sort(arr: List[Union[int, float]]) -> List[Union[int, float]]:
    """Sorts a list using the Gnome Sort algorithm."""
    n = len(arr)
    index = 0
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
    """Sorts a list of integers using the Pigeonhole Sort algorithm."""
    if not all(isinstance(x, int) for x in arr):
        raise TypeError("Pigeonhole sort requires a list of integers.")
    if not arr:
        return []

    min_val, max_val = min(arr), max(arr)
    size = max_val - min_val + 1
    holes = [0] * size

    for x in arr:
        holes[x - min_val] += 1

    sorted_arr = []
    for i in range(size):
        sorted_arr.extend([i + min_val] * holes[i])
    return sorted_arr


def stooge_sort(arr: List[Union[int, float]]) -> List[Union[int, float]]:
    """Sorts a list using the Stooge Sort algorithm (recursive)."""
    def _stooge_sort_recursive(arr_slice, l, h):
        if l >= h:
            return
        if arr_slice[l] > arr_slice[h]:
            arr_slice[l], arr_slice[h] = arr_slice[h], arr_slice[l]

        if h - l + 1 > 2:
            t = (h - l + 1) // 3
            _stooge_sort_recursive(arr_slice, l, h - t)
            _stooge_sort_recursive(arr_slice, l + t, h)
            _stooge_sort_recursive(arr_slice, l, h - t)

    arr_copy = arr[:]  # Work on a copy
    _stooge_sort_recursive(arr_copy, 0, len(arr_copy) - 1)
    return arr_copy


# =============================================================================
# 9. TIME UTILITIES
# =============================================================================

def get_time_difference(
    h1: int, m1: int, h2: int, m2: int
) -> Tuple[int, int]:
    """Calculates the difference in hours and minutes between two times."""
    t1_mins = h1 * 60 + m1
    t2_mins = h2 * 60 + m2
    diff_mins = abs(t2_mins - t1_mins)
    
    hours = diff_mins // 60
    minutes = diff_mins % 60
    return hours, minutes


def convert_12h_to_24h(time_12h: str) -> str:
    """Converts a time string from 12-hour (e.g., "08:05:45 PM") to 24-hour format."""
    is_pm = time_12h[-2:].upper() == "PM"
    time_parts = list(map(int, time_12h[:-2].strip().split(':')))
    hour, minute, second = time_parts[0], time_parts[1], time_parts[2]

    if is_pm and hour != 12:
        hour += 12
    elif not is_pm and hour == 12:  # Midnight case
        hour = 0
        
    return f"{hour:02d}:{minute:02d}:{second:02d}"


def get_angle_between_clock_hands(hour: int, minute: int) -> float:
    """Calculates the smaller angle between the hour and minute hands of a clock."""
    if not (0 <= hour <= 12 and 0 <= minute < 60):
        raise ValueError("Invalid hour or minute.")
        
    # Position of hour hand: (h % 12 + m/60) * 30 degrees
    hour_angle = (hour % 12 + minute / 60) * 30
    # Position of minute hand: m * 6 degrees
    minute_angle = minute * 6
    
    angle = abs(hour_angle - minute_angle)
    # Return the smaller of the two possible angles
    return min(angle, 360 - angle)


# =============================================================================
# 10. DECORATORS & HIGHER-ORDER FUNCTIONS
# =============================================================================

F = TypeVar('F', bound=Callable[..., Any])

def call_counter(func: F) -> F:
    """A decorator that counts and prints the number of times a function is called."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        wrapper.calls += 1
        print(f"Call {wrapper.calls} of {func.__name__!r}")
        return func(*args, **kwargs)
    wrapper.calls = 0
    return wrapper


def timer(func: F) -> F:
    """A decorator that prints the execution time of a function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"{func.__name__!r} executed in {end_time - start_time:.4f}s")
        return result
    return wrapper


# =============================================================================
# 11. EXAMPLE USAGE (MAIN BLOCK)
# =============================================================================

if __name__ == "__main__":
    print("--- Running Examples for Refactored Code ---")

    # --- Geometry Example ---
    p1 = (1, 1)
    p2 = (4, 5)
    distance = calculate_cartesian_distance(p1, p2)
    print(f"\nDistance between {p1} and {p2} is: {distance:.2f}")

    # --- Algebra Example ---
    a, b, c = 2, -11, 5
    print(f"\nQuadratic equation: {a}x^2 + {b}x + {c} = 0")
    print(f"Root type: {get_quadratic_root_type(a, b, c)}")
    print(f"Sum of roots: {get_sum_of_quadratic_roots(a, b, c)}")
    print(f"Product of roots: {get_product_of_quadratic_roots(a, b, c)}")
    roots = calculate_real_quadratic_roots(a, b, c)
    if roots:
        print(f"Roots are: {roots[0]:.2f} and {roots[1]:.2f}")

    # --- Physics Example ---
    mass_of_earth = 5.972e24  # kg
    mass_of_moon = 7.347e22   # kg
    dist_earth_moon = 3.844e8 # meters
    force = calculate_gravitational_force(mass_of_earth, mass_of_moon, dist_earth_moon)
    print(f"\nGravitational force between Earth and Moon: {force:.2e} Newtons")

    # --- Data Structures Example ---
    matrix_a = [["this", "is"], ["program", "for"], ["vertical", "concatenation"]]
    concatenated = concatenate_matrix_columns(matrix_a)
    print(f"\nVertically concatenated matrix: {concatenated}")

    nested_list = [[1, 2, 3], [4, [5, 6]], 7]
    total_sum = sum_nested_list_recursive(nested_list)
    print(f"Sum of nested list {nested_list} is: {total_sum}")
    
    # --- Sorting Example ---
    unsorted_list = [34, 2, 10, -9, 2, 8]
    gnome_sorted = gnome_sort(unsorted_list.copy())
    print(f"\nGnome sorted list: {gnome_sorted}")
    pigeonhole_sorted = pigeonhole_sort([8, 3, 2, 7, 4, 6, 8])
    print(f"Pigeonhole sorted list: {pigeonhole_sorted}")

    # --- Decorator Example ---
    @timer
    @call_counter
    def my_example_function(n):
        """An example function to test decorators."""
        time.sleep(0.01)
        return sum(i for i in range(n))

    print("\n--- Testing Decorators ---")
    my_example_function(1000)
    my_example_function(2000)

    # --- Interactive Examples (commented out by default) ---
    # print("\n--- Interactive List Addition (map) ---")
    # try:
    #     list1_str = input("Enter a list of numbers (space-separated): ")
    #     list2_str = input("Enter a second list of numbers (space-separated): ")
    #     list1 = [int(i) for i in list1_str.split()]
    #     list2 = [int(i) for i in list2_str.split()]
    #     result = list(map(operator.add, list1, list2))
    #     print(f"Sum of lists: {result}")
    # except (ValueError, IndexError):
    #     print("Invalid input. Please enter space-separated numbers.")