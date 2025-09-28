"""
Utilities for cryptographic and hashing operations.
"""
import hashlib
import secrets
from pathlib import Path
from typing import Union

from cryptography.fernet import Fernet

# --- Encryption/Decryption Class ---

class Secure:
    """
    A class to handle symmetric encryption and decryption of messages using Fernet.
    """
    def __init__(self, key_path: Union[str, Path] = "secret.key"):
        """
        Initializes the Secure instance.

        If the key file does not exist, it generates a new key and saves it.
        Otherwise, it loads the existing key.

        Args:
            key_path (Union[str, Path]): The file path to save/load the encryption key.
        """
        self.key_path = Path(key_path)
        if not self.key_path.exists():
            self._generate_and_save_key()
        self.key = self._load_key()
        self.fernet = Fernet(self.key)

    def _generate_and_save_key(self) -> None:
        """Generates a new key and saves it to the specified file."""
        key = Fernet.generate_key()
        with open(self.key_path, "wb") as key_file:
            key_file.write(key)
        print(f"New key generated and saved to '{self.key_path}'")

    def _load_key(self) -> bytes:
        """Loads the encryption key from the file."""
        try:
            with open(self.key_path, "rb") as key_file:
                return key_file.read()
        except FileNotFoundError:
            print(f"Error: Key file not found at '{self.key_path}'")
            raise

    def encrypt_message(self, message: str) -> bytes:
        """
        Encrypts a string message.

        Args:
            message (str): The plaintext message to encrypt.

        Returns:
            bytes: The encrypted message.
        """
        encoded_message = message.encode('utf-8')
        return self.fernet.encrypt(encoded_message)

    def decrypt_message(self, encrypted_message: bytes) -> str:
        """
        Decrypts an encrypted message.

        Args:
            encrypted_message (bytes): The encrypted message to decrypt.

        Returns:
            str: The decrypted plaintext message.
        """
        decrypted_message = self.fernet.decrypt(encrypted_message)
        return decrypted_message.decode('utf-8')


# --- Hashing Functions ---

def get_sha256(text: str) -> str:
    """
    Generates a SHA256 hash for the given text.

    Args:
        text (str): The input string to hash.

    Returns:
        str: The hexadecimal SHA256 hash string.
    """
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def check_sha256_hash(hashed_text: str, data: str) -> bool:
    """
    Verifies if a given SHA256 hash matches the hash of the data.

    Args:
        hashed_text (str): The existing hash to check against.
        data (str): The plaintext data to hash and compare.

    Returns:
        bool: True if the hashes match, False otherwise.
    """
    return hashed_text == get_sha256(data)

# --- Secure Token Generation ---

def get_cryptographically_secure_data(num_bytes: int = 32) -> tuple[bytes, str]:
    """
    Generates cryptographically strong pseudo-random data.

    Args:
        num_bytes (int): The number of random bytes to generate.

    Returns:
        tuple[bytes, str]: A tuple containing the random bytes and its hex representation.
    """
    token_bytes = secrets.token_bytes(num_bytes)
    token_hex = secrets.token_hex(num_bytes)
    return token_bytes, token_hex


"""
Utilities for fetching data from web APIs and URLs.
"""
import time
from typing import Dict, Any, List, Optional
import requests
import yfinance as yf
import pandas as pd


def get_html_content(url: str) -> Optional[str]:
    """
    Retrieves the HTML content of a given URL.

    Args:
        url (str): The URL to fetch.

    Returns:
        Optional[str]: The decoded HTML content as a string, or None if an error occurs.
    """
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching URL {url}: {e}")
        return None


def fetch_btc_price() -> Optional[float]:
    """
    Fetches the latest Bitcoin price from the Bitstamp API.

    Returns:
        Optional[float]: The last price of Bitcoin in USD, or None if an error occurs.
    """
    url = "https://www.bitstamp.net/api/v2/ticker/btcusd/"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return float(data["last"])
    except requests.exceptions.RequestException as e:
        print(f"Error querying Bitstamp API: {e}")
    except (KeyError, ValueError) as e:
        print(f"Error parsing API response: {e}")
    return None


def get_stock_history(
    ticker: str, start_date: str = '2015-01-01', end_date: str = '2020-12-31'
) -> Optional[pd.DataFrame]:
    """
    Gets historical stock price data for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (e.g., 'TSLA', 'AAPL').
        start_date (str): The start date for the data in 'YYYY-MM-DD' format.
        end_date (str): The end date for the data in 'YYYY-MM-DD' format.

    Returns:
        Optional[pd.DataFrame]: A pandas DataFrame with historical stock data, or None.
    """
    try:
        ticker_data = yf.Ticker(ticker)
        history_df = ticker_data.history(start=start_date, end=end_date)
        if history_df.empty:
            print(f"No data found for ticker '{ticker}' in the given date range.")
            return None
        return history_df
    except Exception as e:
        print(f"An error occurred while fetching stock data: {e}")
        return None


def get_itunes_top_artists(limit: int = 10) -> Optional[List[str]]:
    """
    Gets a list of the top artists from the Apple iTunes top songs RSS feed.

    Args:
        limit (int): The number of top artists to retrieve.

    Returns:
        Optional[List[str]]: A list of artist names, or None if an error occurs.
    """
    url = f'https://itunes.apple.com/us/rss/topsongs/limit={limit}/json'
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        artists = [
            entry['im:artist']['label']
            for entry in data['feed']['entry']
        ]
        return artists
    except requests.exceptions.RequestException as e:
        print(f"Error fetching iTunes data: {e}")
    except (KeyError, TypeError) as e:
        print(f"Error parsing iTunes JSON response: {e}")
    return None



"""
Utilities for data processing, including text analysis and data structure manipulation.
"""
import os
from typing import List, Dict, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt


def get_prominent_words(corpus: List[str], new_docs: List[str], top_n: int = 2) -> List[np.ndarray]:
    """
    Extracts the most prominent words from new documents based on a corpus using TF-IDF.

    Args:
        corpus (List[str]): A list of documents to build the TF-IDF vocabulary.
        new_docs (List[str]): A list of new documents to analyze.
        top_n (int): The number of top words to extract from each new document.

    Returns:
        List[np.ndarray]: A list where each element is an array of top words for a document.
    """
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf.fit(corpus)
    feature_names = np.array(tfidf.get_feature_names_out())
    
    responses = tfidf.transform(new_docs)

    def get_top_tf_idf_words(response, top: int):
        # Sort the indices of the non-zero elements by their value in descending order
        sorted_indices = np.argsort(response.data)[::-1]
        top_indices = response.indices[sorted_indices[:top]]
        return feature_names[top_indices]

    return [get_top_tf_idf_words(response, top_n) for response in responses]


def generate_wordcloud_from_text(text: str) -> None:
    """
    Generates and displays a word cloud from a given string.

    Args:
        text (str): The input text to visualize.
    """
    stopwords = set(STOPWORDS)
    wordcloud = WordCloud(
        width=800, height=800,
        background_color='white',
        stopwords=stopwords,
        min_font_size=10
    ).generate(text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.show()

def generate_wordcloud_from_file(filepath: str) -> None:
    """
    Generates and displays a word cloud from a text file.

    Args:
        filepath (str): The path to the text file.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        generate_wordcloud_from_text(text)
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'")
    except Exception as e:
        print(f"An error occurred: {e}")


def sort_list_of_dicts(
    data: List[Dict[str, Any]], sort_key: str, reverse: bool = False
) -> List[Dict[str, Any]]:
    """
    Sorts a list of dictionaries by a specified key.

    Args:
        data (List[Dict[str, Any]]): The list of dictionaries to sort.
        sort_key (str): The key to sort by.
        reverse (bool): If True, sorts in descending order.

    Returns:
        List[Dict[str, Any]]: The sorted list.
    """
    return sorted(data, key=lambda x: x[sort_key], reverse=reverse)


def sort_list_of_tuples(data: List[Tuple], sort_index: int = -1) -> List[Tuple]:
    """
    Sorts a list of tuples by an element at a specified index.

    Args:
        data (List[Tuple]): The list of tuples to sort.
        sort_index (int): The index of the element to sort by (default is the last element).

    Returns:
        List[Tuple]: The sorted list of tuples.
    """
    return sorted(data, key=lambda x: x[sort_index])


"""
A collection of mathematical utility functions.
"""
import math
from typing import List, Union

Numeric = Union[int, float]

def rectangle_area(length: Numeric, breadth: Numeric) -> Numeric:
    """Calculates the area of a rectangle."""
    if length < 0 or breadth < 0:
        raise ValueError("Length and breadth must be non-negative.")
    return length * breadth

def triangle_area_heron(s1: Numeric, s2: Numeric, s3: Numeric) -> float:
    """Calculates the area of a triangle using Heron's formula."""
    if not (s1 + s2 > s3 and s1 + s3 > s2 and s2 + s3 > s1):
        raise ValueError("The given side lengths do not form a valid triangle.")
    s = (s1 + s2 + s3) / 2
    area = math.sqrt(s * (s - s1) * (s - s2) * (s - s3))
    return area

def simple_interest(principal: Numeric, rate: Numeric, time: Numeric) -> float:
    """Calculates simple interest."""
    if principal < 0 or rate < 0 or time < 0:
        raise ValueError("Principal, rate, and time must be non-negative.")
    return (principal * rate * time) / 100

def compound_interest(principal: Numeric, rate: Numeric, time: Numeric) -> float:
    """Calculates compound interest."""
    if principal < 0 or rate < 0 or time < 0:
        raise ValueError("Principal, rate, and time must be non-negative.")
    amount = principal * (pow((1 + rate / 100), time))
    return amount - principal

def factorial(n: int) -> int:
    """Calculates the factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers.")
    return 1 if (n == 1 or n == 0) else n * factorial(n - 1)

def circle_area(radius: Numeric) -> float:
    """Calculates the area of a circle."""
    if radius < 0:
        raise ValueError("Radius must be non-negative.")
    return math.pi * (radius ** 2)

def is_prime(num: int) -> bool:
    """Checks if a number is a prime number."""
    if num <= 1:
        return False
    for i in range(2, int(math.sqrt(num)) + 1):
        if num % i == 0:
            return False
    return True

def get_divisors(num: int) -> List[int]:
    """Finds all divisors of an integer."""
    if num == 0:
        return []
    divs = [i for i in range(1, abs(num) // 2 + 1) if num % i == 0]
    divs.append(abs(num))
    return divs

def is_leap_year(year: int) -> bool:
    """Determines if a year is a leap year."""
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)

def calculate_mean(numbers: List[Numeric]) -> float:
    """Calculates the mean (average) of a list of numbers."""
    if not numbers:
        return 0.0
    return sum(numbers) / len(numbers)

def calculate_std_dev(numbers: List[Numeric]) -> float:
    """Calculates the standard deviation of a list of numbers."""
    if len(numbers) < 2:
        return 0.0
    mean = calculate_mean(numbers)
    variance = sum([(x - mean) ** 2 for x in numbers]) / len(numbers)
    return math.sqrt(variance)



"""
A collection of general-purpose utility functions.
"""
import uuid
from typing import List, Any, Generator, Tuple

def generate_uuid() -> uuid.UUID:
    """Generates a Unique Universal Identifier (UUID4)."""
    return uuid.uuid4()

def infinite_sequence() -> Generator[int, None, None]:
    """A generator for an infinite sequence of integers starting from 0."""
    n = 0
    while True:
        yield n
        n += 1

def cm_to_feet_inches(cm: float) -> Tuple[int, float]:
    """Converts centimeters to feet and inches."""
    if cm < 0:
        raise ValueError("Height in centimeters must be non-negative.")
    inches_total = cm * 0.393701
    feet = int(inches_total // 12)
    remaining_inches = inches_total % 12
    return feet, remaining_inches

def reverse_number(n: int) -> int:
    """Reverses the digits of an integer."""
    is_negative = n < 0
    n = abs(n)
    rev = 0
    while n > 0:
        remainder = n % 10
        rev = (rev * 10) + remainder
        n = n // 10
    return -rev if is_negative else rev

def is_palindrome(s: str) -> bool:
    """Checks if a string is a palindrome (reads the same forwards and backwards)."""
    normalized_s = ''.join(filter(str.isalnum, s)).lower()
    return normalized_s == normalized_s[::-1]

def find_common_elements(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Finds common elements between two lists."""
    return list(set(list1) & set(list2))

def remove_duplicates_from_list(data: List[Any]) -> List[Any]:
    """Removes duplicate elements from a list while preserving order."""
    return list(dict.fromkeys(data))



"""
Main script to demonstrate the functionality of the utility modules.
"""
import time
import general_utils
import math_utils
import security_utils
import web_utils
import data_processing

def demonstrate_security():
    """Shows examples from the security_utils module."""
    print("\n--- Security Utilities ---")
    
    # Encryption/Decryption
    secure_handler = security_utils.Secure("my_app.key")
    secret_text = "This is a very confidential message."
    encrypted = secure_handler.encrypt_message(secret_text)
    print(f"Original message: {secret_text}")
    print(f"Encrypted message: {encrypted}")
    decrypted = secure_handler.decrypt_message(encrypted)
    print(f"Decrypted message: {decrypted}")

    # Hashing
    hashed = security_utils.get_sha256("Python is fun!")
    print(f"\nSHA256 Hash: {hashed}")
    is_valid = security_utils.check_sha256_hash(hashed, "Python is fun!")
    print(f"Is hash valid? {is_valid}")

def demonstrate_web():
    """Shows examples from the web_utils module."""
    print("\n--- Web Utilities ---")
    
    # Get Bitcoin Price
    btc_price = web_utils.fetch_btc_price()
    if btc_price:
        print(f"Current Bitcoin Price: ${btc_price:,.2f} USD")

    # Get Stock Data
    tesla_stock = web_utils.get_stock_history("TSLA", "2020-01-01", "2020-12-31")
    if tesla_stock is not None:
        print("\nTesla (TSLA) Stock Data (first 5 rows of 2020):")
        print(tesla_stock.head())
        
    # Get Top Artists
    top_artists = web_utils.get_itunes_top_artists(5)
    if top_artists:
        print("\nTop 5 Artists on iTunes:")
        for i, artist in enumerate(top_artists, 1):
            print(f"{i}. {artist}")

def demonstrate_math():
    """Shows examples from the math_utils module."""
    print("\n--- Math Utilities ---")
    
    # Area calculation
    print(f"Area of 5x4 rectangle: {math_utils.rectangle_area(5, 4)}")
    print(f"Area of triangle (3, 4, 5): {math_utils.triangle_area_heron(3, 4, 5):.2f}")
    
    # Prime number check
    num = 29
    print(f"Is {num} a prime number? {math_utils.is_prime(num)}")
    
    # Leap year check
    year = 2024
    print(f"Is {year} a leap year? {math_utils.is_leap_year(year)}")

def demonstrate_data_processing():
    """Shows examples from the data_processing module."""
    print("\n--- Data Processing Utilities ---")

    # Sort list of dictionaries
    animals = [
        {'type': 'lion', 'name': 'Mr. T', 'age': 7},
        {'type': 'tiger', 'name': 'scarface', 'age': 3},
        {'type': 'puma', 'name': 'Joe', 'age': 4}
    ]
    sorted_animals = data_processing.sort_list_of_dicts(animals, 'age', reverse=True)
    print("Animals sorted by age (descending):")
    print(sorted_animals)

    # WordCloud from text (requires a file to exist)
    # Example: Create a file 'sample.txt' with some text.
    # data_processing.generate_wordcloud_from_file('sample.txt')

def main():
    """Main function to run all demonstrations."""
    demonstrate_security()
    demonstrate_web()
    demonstrate_math()
    demonstrate_data_processing()
    
    # Interactive example
    print("\n--- Interactive Example: Height Converter ---")
    try:
        cm_input = input("Enter your height in centimeters: ")
        cm = float(cm_input)
        feet, inches = general_utils.cm_to_feet_inches(cm)
        print(f"{cm} cm is approximately {feet} feet and {inches:.2f} inches.")
    except ValueError:
        print("Invalid input. Please enter a number.")

if __name__ == "__main__":
    main()
