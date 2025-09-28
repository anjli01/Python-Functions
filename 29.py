# -*- coding: utf-8 -*-
"""
A collection of refactored Python code examples covering various domains
including physics, data structures, algorithms, and general utilities.

This script follows standard Python best practices, including PEP 8 styling,
docstrings, type hinting, and separation of concerns.

Team Members: Santu Hazra, Manu Chauhan, Ammar Adil, and Prakash Nishtala.
Refactored by an AI assistant following standard practices.
"""

# =============================================================================
# 1. IMPORTS
# =============================================================================
# Standard Library Imports
import hashlib
import math
import os
import secrets
import string
import uuid
from collections import Counter, deque
from functools import reduce
from html.parser import HTMLParser
from itertools import (
    combinations,
    combinations_with_replacement,
    groupby,
    permutations,
)
from typing import (
    Any,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)
import urllib.request
import re
import random
from time import sleep

# Third-party Library Imports
# Note: These libraries need to be installed via pip.
# pip install nltk cryptography requests yfinance scikit-learn wordcloud matplotlib
try:
    import nltk
    from cryptography.fernet import Fernet
    import requests
    import yfinance as yf
    from sklearn.feature_extraction.text import TfidfVectorizer
    import numpy as np
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
except ImportError as e:
    print(f"Warning: A required third-party library is not installed. {e}")
    print("Please run: pip install nltk cryptography requests yfinance scikit-learn wordcloud matplotlib numpy")


# =============================================================================
# 2. CONSTANTS
# =============================================================================
COULOMB_CONSTANT: float = 8.9875517923e9  # More precise value for k
PI: float = math.pi
LETTERS_LOWERCASE: str = string.ascii_lowercase


# =============================================================================
# 3. UTILITY & SCIENTIFIC FUNCTIONS
# =============================================================================

def calculate_electrostatic_force(q1: float, q2: float, distance: float) -> float:
    """Calculates the electrostatic force between two charged particles.

    Args:
        q1: Charge of the first particle in Coulombs.
        q2: Charge of the second particle in Coulombs.
        distance: Distance between the particles in meters.

    Returns:
        The electrostatic force in Newtons.

    Raises:
        ValueError: If the distance is zero.
    """
    if distance == 0:
        raise ValueError("Distance cannot be zero.")
    return (COULOMB_CONSTANT * q1 * q2) / (distance**2)


def calculate_density(mass: float, volume: float) -> float:
    """Calculates the density of a substance.

    Args:
        mass: Mass of the substance.
        volume: Volume of the substance.

    Returns:
        The density (mass/volume).

    Raises:
        ValueError: If the volume is zero.
    """
    if volume == 0:
        raise ValueError("Volume cannot be zero.")
    return mass / volume


def convert_temperature(temp: float, unit: str = 'c') -> float:
    """Converts temperature between Celsius and Fahrenheit.

    Args:
        temp: The temperature value to convert.
        unit: The unit of the input temperature ('c' for Celsius, 'f' for Fahrenheit).

    Returns:
        The converted temperature.
    """
    unit = unit.lower()
    if unit == 'c':  # Convert Celsius to Fahrenheit
        return (temp * 9/5) + 32
    elif unit == 'f':  # Convert Fahrenheit to Celsius
        return (temp - 32) * (5/9)
    else:
        raise ValueError("Invalid unit. Please use 'c' for Celsius or 'f' for Fahrenheit.")


def calculate_power_recursive(base: Union[int, float], exponent: int) -> Union[int, float]:
    """Calculates the power of a number using recursion.

    Args:
        base: The base number (N).
        exponent: The exponent (P), must be a non-negative integer.

    Returns:
        The result of base raised to the power of exponent.
    """
    if exponent < 0:
        raise ValueError("Exponent must be a non-negative integer.")
    if exponent == 0:
        return 1
    if exponent == 1:
        return base
    return base * calculate_power_recursive(base, exponent - 1)


def calculate_time_difference(h1: int, m1: int, h2: int, m2: int) -> Tuple[int, int]:
    """Calculates the difference between two times.

    Args:
        h1: Hour of the first time (0-23).
        m1: Minute of the first time (0-59).
        h2: Hour of the second time (0-23).
        m2: Minute of the second time (0-59).

    Returns:
        A tuple (hours, minutes) representing the difference.
    """
    t1_minutes = h1 * 60 + m1
    t2_minutes = h2 * 60 + m2
    diff_minutes = abs(t2_minutes - t1_minutes)
    
    hours = diff_minutes // 60
    minutes = diff_minutes % 60
    return hours, minutes


def convert_12h_to_24h_format(time_12h: str) -> str:
    """Converts time from 12-hour format (e.g., "08:05:45 PM") to 24-hour format.

    Args:
        time_12h: A string representing time in 12-hour format.

    Returns:
        A string representing the time in 24-hour format.
    """
    from datetime import datetime
    # The format code %I is for 12-hour clock, %p for AM/PM.
    in_time = datetime.strptime(time_12h, "%I:%M:%S %p")
    # %H is for 24-hour clock.
    return in_time.strftime("%H:%M:%S")


def calculate_clock_angle(hour: int, minute: int) -> float:
    """Calculates the smaller angle between the hour and minute hands of a clock.

    Args:
        hour: The hour (0-12).
        minute: The minute (0-59).

    Returns:
        The smaller angle in degrees.
    """
    if not (0 <= hour <= 12 and 0 <= minute <= 59):
        raise ValueError("Invalid time input.")
    
    hour_angle = 0.5 * (hour * 60 + minute)
    minute_angle = 6 * minute
    
    angle = abs(hour_angle - minute_angle)
    return min(360 - angle, angle)


def find_first_time_for_angle(target_angle: float) -> Optional[str]:
    """Finds the first time on a clock where the hands have a specific angle.

    Args:
        target_angle: The desired angle in degrees.

    Returns:
        The time as a formatted string "HH:MM", or None if not found.
    """
    for h in range(12):
        for m in range(60):
            # Use math.isclose for floating point comparison
            if math.isclose(calculate_clock_angle(h, m), target_angle, rel_tol=1e-5):
                return f"{h:02d}:{m:02d}"
    return None


# =============================================================================
# 4. STRING MANIPULATION FUNCTIONS
# =============================================================================

def remove_punctuation(text: str) -> str:
    """Removes all punctuation from a string.

    Args:
        text: The input string.

    Returns:
        The string with punctuation removed.
    """
    return text.translate(str.maketrans('', '', string.punctuation))


def filter_strings_by_substring(
    data: List[str], substring: str, start: int, end: int
) -> List[str]:
    """Filters a list of strings to find those containing a substring at a specific position.

    Args:
        data: A list of strings to filter.
        substring: The substring to search for.
        start: The starting index of the slice.
        end: The ending index of the slice.

    Returns:
        A new list containing only the matching strings.
    """
    return [s for s in data if s[start:end] == substring]


def swap_case(text: str) -> str:
    """Swaps the case of all letters in a string.

    Args:
        text: The input string.

    Returns:
        The string with the case of all letters swapped.
    """
    return text.swapcase()


def compress_string(text: str) -> str:
    """Compresses a string by replacing consecutive characters with (count, char).
    
    Example: 'AAABBC' -> '(3, A) (2, B) (1, C)'

    Args:
        text: The string to compress.

    Returns:
        A formatted string representing the compressed data.
    """
    if not text:
        return ""
    
    groups = [f"({len(list(g))}, {k})" for k, g in groupby(text)]
    return " ".join(groups)

def count_repeated_substring_in_string(s: str, n: int, sub: str = 'a') -> int:
    """Counts occurrences of a substring in a string repeated n times.

    Args:
        s: The base string.
        n: The total length of the final repeated string.
        sub: The substring to count (defaults to 'a').

    Returns:
        The total count of the substring.
    """
    len_s = len(s)
    if len_s == 0:
        return 0
    
    count_in_s = s.count(sub)
    num_repeats = n // len_s
    remainder_len = n % len_s
    
    total_count = count_in_s * num_repeats + s[:remainder_len].count(sub)
    return total_count


def find_vowel_substrings_between_consonants(text: str) -> List[str]:
    """Finds all substrings that contain 2+ vowels between two consonants.

    Args:
        text: The input string to search.

    Returns:
        A list of matching vowel substrings, or an empty list if none are found.
    """
    vowels = "aeiou"
    consonants = "qwrtypsdfghjklzxcvbnm"
    # Case-insensitive search using re.I
    pattern = f"(?<=[{consonants}])([{vowels}]{{2,}})(?=[{consonants}])"
    return re.findall(pattern, text, flags=re.I)


# =============================================================================
# 5. DATA STRUCTURE & ALGORITHM FUNCTIONS
# =============================================================================

def merge_dictionary_lists(list1: List[Dict], list2: List[Dict]) -> List[Dict]:
    """Merges two lists of dictionaries element-wise.

    For each index `i`, it merges `list2[i]` into `list1[i]`, adding new keys
    from `list2` without overwriting existing keys in `list1`.
    Assumes lists are of the same length.

    Args:
        list1: The base list of dictionaries.
        list2: The list of dictionaries to merge from.

    Returns:
        The modified first list with merged dictionaries.
    """
    if len(list1) != len(list2):
        raise ValueError("Input lists must be of the same length.")
    
    for i, dict1 in enumerate(list1):
        dict2 = list2[i]
        for key, value in dict2.items():
            if key not in dict1:
                dict1[key] = value
    return list1


def get_kth_column(matrix: List[List[Any]], k: int) -> Tuple:
    """Extracts the Kth column from a matrix.

    Args:
        matrix: A 2D list representing the matrix.
        k: The zero-based index of the column to extract.

    Returns:
        A tuple containing the elements of the Kth column.
    
    Raises:
        IndexError: If k is out of bounds.
    """
    if not matrix or k >= len(matrix[0]):
        raise IndexError("Column index k is out of bounds.")
    
    # The zip(*matrix) trick is a Pythonic way to transpose a matrix
    return tuple(zip(*matrix))[k]


def get_all_subarrays(arr: List[Any]) -> List[List[Any]]:
    """Generates all possible contiguous subarrays of a list using recursion.

    Args:
        arr: The input list.

    Returns:
        A list containing all subarrays.
    """
    subarrays = []
    
    def _generate(start: int, end: int):
        if end == len(arr):
            return
        if start > end:
            _generate(0, end + 1)
        else:
            subarrays.append(arr[start:end + 1])
            _generate(start + 1, end)
            
    _generate(0, 0)
    return subarrays


def sum_nested_list(data: List) -> Union[int, float]:
    """Calculates the sum of all numbers in a potentially nested list.

    This function is refactored to avoid global variables.

    Args:
        data: A list that may contain numbers or other lists.

    Returns:
        The total sum of all numbers in the list and its sublists.
    """
    total = 0
    for item in data:
        if isinstance(item, list):
            total += sum_nested_list(item)
        elif isinstance(item, (int, float)):
            total += item
    return total

# --- Sorting Algorithms ---

def gnome_sort(arr: List[Union[int, float]]) -> None:
    """Sorts a list in-place using the Gnome Sort algorithm.

    Time Complexity: O(n^2)
    Space Complexity: O(1)

    Args:
        arr: The list of numbers to be sorted.
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


def pigeonhole_sort(arr: List[int]) -> None:
    """Sorts a list of integers in-place using the Pigeonhole Sort algorithm.

    Suitable for lists where the number of elements and the range of values are similar.

    Args:
        arr: The list of integers to be sorted.
    """
    if not arr:
        return
        
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


def stooge_sort(arr: List[Union[int, float]], l: int, h: int) -> None:
    """Sorts a list in-place using the recursive Stooge Sort algorithm.

    Args:
        arr: The list to be sorted.
        l: The starting index.
        h: The ending index.
    """
    if l >= h:
        return

    if arr[l] > arr[h]:
        arr[l], arr[h] = arr[h], arr[l]

    if h - l + 1 > 2:
        t = (h - l + 1) // 3
        stooge_sort(arr, l, h - t)
        stooge_sort(arr, l + t, h)
        stooge_sort(arr, l, h - t)


def divisible_sum_pairs(arr: List[int], k: int) -> int:
    """Finds the number of pairs (i, j) where i < j and arr[i] + arr[j] is divisible by k.

    Args:
        arr: A list of integers.
        k: The divisor.

    Returns:
        The count of such pairs.
    """
    count = 0
    n = len(arr)
    for i in range(n):
        for j in range(i + 1, n):
            if (arr[i] + arr[j]) % k == 0:
                count += 1
    return count

# ... (other sorting algorithms like bubble, selection, etc. can be added here with similar refactoring)


# =============================================================================
# 6. FILE I/O FUNCTIONS
# =============================================================================

def count_words_in_file(filepath: str) -> int:
    """Counts the total number of words in a text file.

    Args:
        filepath: The path to the text file.

    Returns:
        The total word count.
    
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: '{filepath}'")
    
    word_count = 0
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            word_count += len(line.split())
    return word_count


# =============================================================================
# 7. CLASSES
# =============================================================================

class Circle:
    """Represents a circle with a given radius."""
    def __init__(self, radius: float):
        """Initializes the Circle.

        Args:
            radius: The radius of the circle.

        Raises:
            ValueError: If the radius is negative.
        """
        if radius < 0:
            raise ValueError("Radius cannot be negative.")
        self.radius = radius

    def area(self) -> float:
        """Calculates the area of the circle."""
        return PI * self.radius ** 2
        
    def perimeter(self) -> float:
        """Calculates the perimeter (circumference) of the circle."""
        return 2 * PI * self.radius


class Secure:
    """A class to handle encryption and decryption of messages using Fernet."""

    def __init__(self, key_path: str = "secret.key"):
        self.key_path = key_path
        self._generate_and_save_key()

    def _generate_and_save_key(self) -> None:
        """Generates a key and saves it to the key_path file."""
        key = Fernet.generate_key()
        with open(self.key_path, "wb") as key_file:
            key_file.write(key)

    def _load_key(self) -> bytes:
        """Loads the key from the key_path file."""
        return open(self.key_path, "rb").read()

    def encrypt_message(self, message: str) -> bytes:
        """Encrypts a string message.

        Args:
            message: The string to encrypt.

        Returns:
            The encrypted message as bytes.
        """
        key = self._load_key()
        f = Fernet(key)
        encrypted_message = f.encrypt(message.encode('utf-8'))
        return encrypted_message

    def decrypt_message(self, encrypted_message: bytes) -> str:
        """Decrypts an encrypted message.

        Args:
            encrypted_message: The encrypted bytes.

        Returns:
            The decrypted message as a string.
        """
        key = self._load_key()
        f = Fernet(key)
        decrypted_message = f.decrypt(encrypted_message)
        return decrypted_message.decode('utf-8')


# =============================================================================
# 8. ADVANCED & EXTERNAL LIBRARY FUNCTIONS
# =============================================================================

def get_sha256_hash(text: str) -> str:
    """Generates the SHA256 hash for a given text.

    Args:
        text: The input string.

    Returns:
        The hexadecimal SHA256 hash string.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def get_html_from_url(url: str) -> str:
    """Fetches the HTML content of a given URL.

    Args:
        url: The URL to fetch.

    Returns:
        The HTML content as a string.
    """
    with urllib.request.urlopen(url) as response:
        html_bytes = response.read()
        return html_bytes.decode('utf-8')


def get_current_bitcoin_price() -> Optional[float]:
    """Fetches the current Bitcoin price in USD from Bitstamp.

    Returns:
        The price as a float, or None if the request fails.
    """
    url = "https://www.bitstamp.net/api/ticker/"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raises an HTTPError for bad responses
        data = response.json()
        return float(data["last"])
    except (requests.RequestException, KeyError, ValueError) as e:
        print(f"Error querying Bitstamp API: {e}")
        return None


def generate_wordcloud_from_text(text: str) -> None:
    """Generates and displays a word cloud from the given text.

    Args:
        text: The source text for the word cloud.
    """
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(width=800, height=800,
                          background_color='white',
                          stopwords=stopwords,
                          min_font_size=10).generate(text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()


# =============================================================================
# 9. MAIN EXECUTION BLOCK
# =============================================================================

def main():
    """Main function to demonstrate the refactored code."""
    print("--- Running Demonstrations for Refactored Code ---")

    # --- Scientific Functions ---
    print("\n[Scientific Functions]")
    force = calculate_electrostatic_force(q1=1.6e-19, q2=-1.6e-19, distance=1e-10)
    print(f"Electrostatic force: {force:.4e} N")
    density = calculate_density(mass=100, volume=10)
    print(f"Density: {density} kg/m^3")
    temp_f = convert_temperature(25, unit='c')
    print(f"25°C is {temp_f:.1f}°F")
    temp_c = convert_temperature(77, unit='f')
    print(f"77°F is {temp_c:.1f}°C")
    power_val = calculate_power_recursive(5, 3)
    print(f"5 to the power of 3 is: {power_val}")

    # --- String Manipulation ---
    print("\n[String Manipulation]")
    clean_str = remove_punctuation("Hello, world! This is a test.")
    print(f"String without punctuation: '{clean_str}'")
    swapped = swap_case("Python Is Fun!")
    print(f"Swapped case string: '{swapped}'")
    compressed = compress_string("AAABBCDDDDE")
    print(f"Compressed string: {compressed}")

    # --- Data Structures & Algorithms ---
    print("\n[Data Structures & Algorithms]")
    nested_list = [[1, 2, 3], [4, [5, 6]], 7]
    total_sum = sum_nested_list(nested_list)
    print(f"Sum of nested list {nested_list} is: {total_sum}")
    
    arr_to_sort = [34, 2, 10, -9, 5]
    print(f"Original array: {arr_to_sort}")
    gnome_sort(arr_to_sort)
    print(f"Array after Gnome Sort: {arr_to_sort}")

    subarrays = get_all_subarrays([1, 2, 3])
    print(f"All subarrays of [1, 2, 3]: {subarrays}")

    # --- Classes ---
    print("\n[Classes]")
    my_circle = Circle(radius=10)
    print(f"Circle with radius 10 -> Area: {my_circle.area():.2f}, Perimeter: {my_circle.perimeter():.2f}")
    
    # Secure encryption demo
    secure_handler = Secure()
    secret_message = "This is a very secret message."
    encrypted = secure_handler.encrypt_message(secret_message)
    print(f"Encrypted message (first 20 bytes): {encrypted[:20]}...")
    decrypted = secure_handler.decrypt_message(encrypted)
    print(f"Decrypted message: '{decrypted}'")
    # Clean up the key file
    if os.path.exists(secure_handler.key_path):
        os.remove(secure_handler.key_path)

    # --- External Library Demos ---
    print("\n[External Library Functions]")
    print("Fetching current Bitcoin price...")
    btc_price = get_current_bitcoin_price()
    if btc_price:
        print(f"Current Bitcoin Price: ${btc_price:,.2f}")
    else:
        print("Could not retrieve Bitcoin price.")
        
    # Word cloud demo (will open a plot window)
    print("\nGenerating a word cloud (a plot window will open)...")
    sample_text = "python python python refactoring code readability standards best practices maintainability"
    try:
        generate_wordcloud_from_text(sample_text)
        print("Word cloud generated successfully.")
    except NameError:
        print("Could not generate word cloud. Ensure matplotlib and wordcloud are installed.")


if __name__ == "__main__":
    main()
