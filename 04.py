# 31. Extract words starting with a vowel from a list
from typing import List

VOWELS = "aeiou"

def extract_vowel_words(word_list: List[str]) -> List[str]:
    """
    Extracts words from a list that start with a vowel.

    Args:
        word_list: A list of strings.

    Returns:
        A list containing only the words that start with a vowel.
    """
    return [word for word in word_list if word and word[0].lower() in VOWELS]

if __name__ == "__main__":
    test_list = ["have", "a", "good", "one", "Apple"]
    print(f"The original list is: {test_list}")
    extracted_words = extract_vowel_words(test_list)
    print(f"The extracted words: {extracted_words}")



# 32. Replace vowels by the next vowel
def replace_vowels_with_next(text: str) -> str:
    """
    Replaces each vowel in a string with the next vowel in the sequence (a -> e, e -> i, ..., u -> a).

    Args:
        text: The input string.

    Returns:
        The string with vowels replaced.
    """
    vowels = 'aeiou'
    vowel_map = dict(zip(vowels, vowels[1:] + vowels[0]))
    return "".join([vowel_map.get(char, char) for char in text])

if __name__ == "__main__":
    test_str = 'helloworld'
    print(f"The original string is: {test_str}")
    replaced_str = replace_vowels_with_next(test_str)
    print(f"The replaced string: {replaced_str}")



# 33. Reverse words of a string
def reverse_words_in_sentence(sentence: str) -> str:
    """
    Reverses the order of words in a sentence.

    Args:
        sentence: The input string sentence.

    Returns:
        A string with the words in reverse order.
    """
    words = sentence.split()
    reversed_sentence = ' '.join(reversed(words))
    return reversed_sentence

if __name__ == "__main__":
    input_sentence = 'have a good day'
    print(f"Original: '{input_sentence}'")
    print(f"Reversed: '{reverse_words_in_sentence(input_sentence)}'")



# 34. Find the least frequent character in a string
from collections import Counter
from typing import Optional

def find_least_frequent_char(text: str) -> Optional[str]:
    """
    Finds the character that appears least frequently in a string.
    If there's a tie, one of the least frequent characters is returned.

    Args:
        text: The input string.

    Returns:
        The least frequent character, or None if the string is empty.
    """
    if not text:
        return None
    
    char_counts = Counter(text)
    # The min function's key argument gets the value (count) for each character
    return min(char_counts, key=char_counts.get)

if __name__ == "__main__":
    test_str = "helloworld"
    print(f"The original string is: {test_str}")
    least_frequent = find_least_frequent_char(test_str)
    print(f"The least frequent character is: '{least_frequent}'")



# 35. Find the most frequent element in a list
from collections import Counter
from typing import List, Any, Optional

def find_most_frequent_element(items: List[Any]) -> Optional[Any]:
    """
    Finds the element that appears most frequently in a list.
    If there's a tie, one of the most frequent elements is returned.

    Args:
        items: A list of elements.

    Returns:
        The most frequent element, or None if the list is empty.
    """
    if not items:
        return None
        
    counts = Counter(items)
    # .most_common(1) returns a list like [('item', count)]
    return counts.most_common(1)[0][0]

if __name__ == "__main__":
    num_list = [2, 1, 2, 2, 1, 3, 2]
    print(f"Original list: {num_list}")
    most_frequent = find_most_frequent_element(num_list)
    print(f"The most frequent element is: {most_frequent}")



# 40. Check if two lists share at least one common element
from typing import List, Any

def have_common_element(list1: List[Any], list2: List[Any]) -> bool:
    """
    Checks if there is at least one common element between two lists.

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        True if they share a common element, False otherwise.
    """
    # Sets provide fast membership testing. isdisjoint() is True if they have no common elements.
    return not set(list1).isdisjoint(set(list2))

if __name__ == "__main__":
    a = [1, 2, 3, 4, 5]
    b = [5, 6, 7, 8, 9]
    c = [6, 7, 8, 9]
    
    print(f"Lists {a} and {b} have common element: {have_common_element(a, b)}")
    print(f"Lists {a} and {c} have common element: {have_common_element(a, c)}")



# 41. Calculate the area of a triangle given its sides (Heron's formula)
import math

def calculate_triangle_area(a: float, b: float, c: float) -> Optional[float]:
    """
    Calculates the area of a triangle using Heron's formula.

    Args:
        a: Length of side a.
        b: Length of side b.
        c: Length of side c.

    Returns:
        The area of the triangle, or None if the sides do not form a valid triangle.
    """
    # Triangle inequality theorem
    if (a + b > c) and (a + c > b) and (b + c > a):
        s = (a + b + c) / 2
        area = math.sqrt(s * (s - a) * (s - b) * (s - c))
        return area
    return None

if __name__ == "__main__":
    # Example with valid sides
    s1, s2, s3 = 5, 6, 7
    area = calculate_triangle_area(s1, s2, s3)
    if area is not None:
        print(f"The area of a triangle with sides {s1}, {s2}, {s3} is {area:.2f}")
    else:
        print(f"The sides {s1}, {s2}, {s3} cannot form a triangle.")

    # Example with invalid sides
    s1, s2, s3 = 1, 2, 10
    area = calculate_triangle_area(s1, s2, s3)
    if area is None:
        print(f"The sides {s1}, {s2}, {s3} cannot form a triangle.")



# 42. Swap two variables
def swap_variables_demo():
    """Demonstrates swapping two variables."""
    try:
        x = int(input('Enter value of x: '))
        y = int(input('Enter value of y: '))
        
        print(f"Original values: x = {x}, y = {y}")
        
        # Pythonic way to swap variables
        x, y = y, x
        
        print(f"Values after swapping: x = {x}, y = {y}")

    except ValueError:
        print("Invalid input. Please enter integers.")

if __name__ == "__main__":
    swap_variables_demo()



# 59. A simple command-line calculator
import operator

def simple_calculator():
    """Runs a simple command-line calculator."""
    
    operations = {
        '1': ('Add', operator.add),
        '2': ('Subtract', operator.sub),
        '3': ('Multiply', operator.mul),
        '4': ('Divide', operator.truediv),
    }

    print("Select operation:")
    for key, (name, _) in operations.items():
        print(f"{key}. {name}")
        
    try:
        choice = input("Enter choice (1/2/3/4): ")
        op_name, operation_func = operations[choice]
        
        num1 = float(input("Enter first number: "))
        num2 = float(input("Enter second number: "))
        
        result = operation_func(num1, num2)
        print(f"Result: {result}")

    except KeyError:
        print("Invalid choice. Please select from 1, 2, 3, or 4.")
    except ValueError:
        print("Invalid input. Please enter valid numbers.")
    except ZeroDivisionError:
        print("Error: Cannot divide by zero.")

if __name__ == "__main__":
    simple_calculator()



# 78. Find the difference between two lists
from typing import List, Any

def get_list_difference(list1: List[Any], list2: List[Any]) -> List[Any]:
    """
    Finds elements that are in one list but not the other (symmetric difference).

    Args:
        list1: The first list.
        list2: The second list.

    Returns:
        A list containing elements unique to each list.
    """
    return list(set(list1).symmetric_difference(set(list2)))

if __name__ == "__main__":
    list1 = [10, 20, 30, 40, 50]
    list2 = [10, 20, 30, 60, 70]

    print(f"list1: {list1}")
    print(f"list2: {list2}")

    # To find elements in list1 but not in list2
    diff1 = list(set(list1) - set(list2))
    print(f"Elements in list1 but not list2: {diff1}")

    # To find elements in list2 but not in list1
    diff2 = list(set(list2) - set(list1))
    print(f"Elements in list2 but not list1: {diff2}")
    
    # To find elements that are in one list or the other, but not both
    sym_diff = get_list_difference(list1, list2)
    print(f"Symmetric Difference (unique elements from both): {sym_diff}")



# 95. Remove leading zeros from an IP address
def remove_leading_zeros_from_ip(ip_address: str) -> str:
    """
    Removes leading zeros from each octet of an IP address.
    Example: "216.08.094.196" -> "216.8.94.196"

    Args:
        ip_address: The input IP address string.

    Returns:
        The IP address with leading zeros removed.
    """
    parts = ip_address.split('.')
    # Converting each part to int and then back to str removes leading zeros.
    normalized_parts = [str(int(part)) for part in parts]
    return ".".join(normalized_parts)

if __name__ == "__main__":
    ip = "216.08.094.196"
    print(f"Original IP: {ip}")
    modified_ip = remove_leading_zeros_from_ip(ip)
    print(f"Modified IP: {modified_ip}")



# 101 (from list). Implement Bubble Sort
from typing import List, TypeVar

# A generic type for comparison
T = TypeVar('T') 

def bubble_sort(items: List[T]) -> List[T]:
    """
    Sorts a list of items in ascending order using the Bubble Sort algorithm.
    This implementation includes an optimization to stop early if the list is already sorted.

    Args:
        items: A list of comparable items.

    Returns:
        A new list containing the sorted items.
    """
    n = len(items)
    # Create a copy to avoid modifying the original list
    sorted_items = items[:] 
    
    for i in range(n - 1):
        swapped = False
        for j in range(0, n - i - 1):
            if sorted_items[j] > sorted_items[j + 1]:
                # Swap the elements
                sorted_items[j], sorted_items[j + 1] = sorted_items[j + 1], sorted_items[j]
                swapped = True
        # If no two elements were swapped by inner loop, then break
        if not swapped:
            break
    return sorted_items

if __name__ == "__main__":
    unsorted_list = [2, 3, 5, 6, 4, 5, 1, 9, 7]
    print(f"Original list: {unsorted_list}")
    sorted_list = bubble_sort(unsorted_list)
    print(f"Sorted list:   {sorted_list}")
