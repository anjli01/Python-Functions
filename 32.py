from typing import List, Any

def is_homogeneous(input_list: List[Any]) -> bool:
    """
    Checks if all elements in a list are of the same type.

    Args:
        input_list: The list to check.

    Returns:
        True if the list is homogeneous, False otherwise. An empty or single-element
        list is considered homogeneous.
    """
    if not input_list:
        return True  # An empty list is homogeneous.
    
    first_element_type = type(input_list[0])
    return all(isinstance(item, first_element_type) for item in input_list[1:])


from typing import List, Any, Type

def filter_by_type(input_list: List[Any], type_to_remove: Type) -> List[Any]:
    """
    Removes all elements of a specified data type from a list.

    Args:
        input_list: The list to filter.
        type_to_remove: The data type to remove (e.g., int, str, float).

    Returns:
        A new list with elements of the specified type removed.
    """
    return [item for item in input_list if not isinstance(item, type_to_remove)]


def firstOccurence(arr, i,j):
  res = 0
  for k in arr:         
      if k == j: 
          break
      if k == i: 
          res += 1
  return res


from typing import List, Any

def count_occurrences_before(sequence: List[Any], target: Any, stop_element: Any) -> int:
    """
    Counts occurrences of a target element in a list before a stop element is found.

    Args:
        sequence: The list to search through.
        target: The element to count.
        stop_element: The element that stops the search.

    Returns:
        The number of times 'target' appears before 'stop_element'.
    """
    count = 0
    for item in sequence:
        if item == stop_element:
            break
        if item == target:
            count += 1
    return count


# Write a program to check whether a file/path/direcory exists or not
file_path = "path/here"
import os.path
os.path.exists(file_path)


import os
from pathlib import Path

def path_exists(path: str) -> bool:
    """
    Checks if a file or directory exists at the given path.
    
    Args:
        path: The file or directory path string.
        
    Returns:
        True if the path exists, False otherwise.
    """
    # Modern approach using pathlib
    return Path(path).exists()
    
    # Traditional approach using os.path
    # return os.path.exists(path)

# Example Usage:
# Note: Create a dummy file or directory to test this.
# with open("test_file.txt", "w") as f:
#     f.write("hello")
#
# print(f"Does 'test_file.txt' exist? {path_exists('test_file.txt')}")
# print(f"Does 'non_existent_file.txt' exist? {path_exists('non_existent_file.txt')}")


# Write a program to merge two python dictionaries
x={'key1':'val1','key2':'val2'}
y={'key3':'val3','key4':'val4'}
z = {**x, **y} # z = x | y  


from typing import Dict, Any

def merge_dictionaries(dict1: Dict[Any, Any], dict2: Dict[Any, Any]) -> Dict[Any, Any]:
    """
    Merges two dictionaries into a new one.
    
    If there are overlapping keys, the value from the second dictionary (dict2) will be used.
    
    Args:
        dict1: The first dictionary.
        dict2: The second dictionary.
        
    Returns:
        A new dictionary containing items from both.
    """
    # For Python 3.5+
    return {**dict1, **dict2}
    
    # For Python 3.9+
    # return dict1 | dict2

# Example Usage:
x = {'key1': 'val1', 'key2': 'val2'}
y = {'key2': 'new_val2', 'key3': 'val3'}
merged_dict = merge_dictionaries(x, y)
print(f"Merged dictionary: {merged_dict}")


def ngcd(x, y):
    i=1
    while(i<=x and i<=y):
        if(x%i==0 and y%i == 0):
            gcd=i
        i+=1
    return gcd
def num_comm_div(x, y):
  n = ngcd(x, y)
  # ... (logic to count divisors of n)


import math

def count_divisors(n: int) -> int:
    """Counts the total number of positive divisors for a given integer."""
    if n <= 0:
        return 0
    
    count = 0
    # Iterate from 1 up to the square root of n
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            # If i is a divisor, n/i is also a divisor.
            if i * i == n:
                count += 1  # Perfect square, count only once.
            else:
                count += 2  # Count both i and n/i.
    return count

def count_common_divisors(a: int, b: int) -> int:
    """
    Finds the number of common divisors between two integers.

    This is equivalent to finding the number of divisors of their
    Greatest Common Divisor (GCD).
    
    Args:
        a: The first integer.
        b: The second integer.
        
    Returns:
        The count of common divisors.
    """
    if a == 0 or b == 0:
        return 0
        
    # Use the highly efficient math.gcd() from the standard library.
    common_divisor = math.gcd(a, b)
    
    return count_divisors(common_divisor)

# Example Usage:
print(f"Number of common divisors between 12 and 18 is: {count_common_divisors(12, 18)}")
# Divisors of 12: 1, 2, 3, 4, 6, 12
# Divisors of 18: 1, 2, 3, 6, 9, 18
# Common: 1, 2, 3, 6 (Count is 4)
# GCD(12, 18) = 6. Divisors of 6 are 1, 2, 3, 6 (Count is 4)


def remove_nums(int_list):
  position = 3 - 1 
  idx = 0
  len_list = (len(int_list))
  while len_list>0:
    idx = (position+idx)%len_list
    print(int_list.pop(idx))
    len_list -= 1


from typing import List, Generator, TypeVar

T = TypeVar('T') # Generic type for list elements

def yield_every_nth(items: List[T], n: int) -> Generator[T, None, None]:
    """
    A generator that yields every nth item from a list, removing them as it goes,
    until the list is empty. This is a variation of the Josephus problem.

    Args:
        items: The list of items to process.
        n: The step count (e.g., 3 for every third item).

    Yields:
        Items from the list in the specified removal order.
    """
    if n <= 0:
        return

    # Work on a copy to avoid modifying the original list passed to the function
    items_copy = list(items)
    index = 0
    
    while items_copy:
        # Calculate the index to remove, wrapping around the list
        index = (index + n - 1) % len(items_copy)
        yield items_copy.pop(index)

# Example Usage:
numbers = [10, 20, 30, 40, 50, 60, 70, 80, 90]
print(f"Removing every 3rd number from {numbers}:")
for removed_number in yield_every_nth(numbers, 3):
    print(removed_number)


string_words = '''...'''
word_list = string_words.split()
word_freq = [word_list.count(n) for n in word_list]
print("Pairs (Words and Frequencies:\n {}".format(str(list(zip(word_list, word_freq)))))


import re
from collections import Counter
from typing import Dict

def count_word_frequency(text: str) -> Dict[str, int]:
    """
    Counts the frequency of each word in a string.
    
    The text is converted to lowercase and punctuation is removed before counting.

    Args:
        text: The string to analyze.

    Returns:
        A dictionary mapping each word to its frequency.
    """
    # Use regex to find all words (sequences of letters) and convert to lowercase.
    # This handles punctuation and makes the count case-insensitive.
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Use collections.Counter for a highly efficient and idiomatic word count.
    return Counter(words)

# Example Usage:
string_words = '''This assignment is of 900 marks. Each example is 9 marks.
If your example is similar to someone else, then you score less.
The formula we will use is 9/(repeated example). That means if 9 people write same example,
then you get only 1. So think different!'''

word_counts = count_word_frequency(string_words)

# Print the most common words for a nice, sorted output
print("Word Frequencies (most common first):")
for word, count in word_counts.most_common():
    print(f"- {word}: {count}")

def permute(nums):
  result_perms = [[]]
  for n in nums:
    new_perms = []
    for perm in result_perms:
      for i in range(len(perm)+1):
        new_perms.append(perm[:i] + [n] + perm[i:])
        result_perms = new_perms
  return result_perms

import itertools
from typing import List, Any

def get_permutations(items: List[Any]) -> List[List[Any]]:
    """
    Generates all possible permutations from a collection of distinct items.

    This implementation uses the standard library's `itertools.permutations`
    for optimal performance and readability.

    Args:
        items: A list of items to permute.

    Returns:
        A list of lists, where each inner list is a unique permutation.
    """
    return [list(p) for p in itertools.permutations(items)]

# Example Usage:
numbers = [1, 2, 3]
permutations = get_permutations(numbers)
print(f"Permutations of {numbers}: {permutations}")


x=10
y=20
z=30
if y < x and x < z:
    print(x)
# ... many more elifs ...



from typing import Union

Numeric = Union[int, float]

def median_of_three(a: Numeric, b: Numeric, c: Numeric) -> Numeric:
    """
    Finds the median value among three numbers.

    Args:
        a: First number.
        b: Second number.
        c: Third number.

    Returns:
        The median of the three numbers.
    """
    # The simplest and most robust way is to sort them and pick the middle element.
    return sorted([a, b, c])[1]

# Example Usage:
x, y, z = 10, 20, 5
print(f"The median of {x}, {y}, and {z} is: {median_of_three(x, y, z)}")



def carry_number(x, y):
  ctr = 0
  if(x == 0 and y == 0):
    return 0
  z = 0  
  for i in reversed(range(10)):
      z = x%10 + y%10 + z
      if z > 9:
        z = 1
      else:
        z = 0
      ctr += z
      x //= 10
      y //= 10
  # ... return logic


def count_carry_operations(a: int, b: int) -> int:
    """
    Counts the number of carry operations when adding two positive integers.

    Args:
        a: The first positive integer.
        b: The second positive integer.

    Returns:
        The total number of carry operations.
    """
    carry = 0
    count = 0
    
    # Ensure a and b are positive
    a, b = abs(a), abs(b)
    
    while a > 0 or b > 0:
        # Get the last digit of each number
        digit_a = a % 10
        digit_b = b % 10
        
        # Check if the sum of digits plus the carry is 10 or more
        if digit_a + digit_b + carry >= 10:
            carry = 1
            count += 1
        else:
            carry = 0
            
        # Move to the next digit
        a //= 10
        b //= 10
        
    return count

# Example Usage:
num1, num2 = 555, 666
#   555
# + 666
# -----
#  1221
# Carries: 1 (5+6), 1 (5+6+1), 1 (5+6+1) -> 3 carries
print(f"Carry operations for {num1} + {num2}: {count_carry_operations(num1, num2)}")
print(f"Carry operations for 123 + 456: {count_carry_operations(123, 456)}")


def volumeTriangular(a, b, h): return (0.1666) * a * b * h 
def volumeSquare(b, h): return (0.33) * b * b * h 
def volumePentagonal(a, b, h): return (0.83) * a * b * h 
def volumeHexagonal(a, b, h): return a * b * h


import math

def pyramid_volume(base_area: float, height: float) -> float:
    """Calculates the volume of any pyramid given its base area and height."""
    return (1/3) * base_area * height

def square_pyramid_volume(base_side: float, height: float) -> float:
    """Calculates the volume of a square pyramid."""
    base_area = base_side ** 2
    return pyramid_volume(base_area, height)

def triangular_pyramid_volume_from_base(base_width: float, base_height: float, pyramid_height: float) -> float:
    """
    Calculates the volume of a triangular pyramid given its base dimensions
    and the pyramid's height.
    """
    base_area = 0.5 * base_width * base_height
    return pyramid_volume(base_area, pyramid_height)

# Note: Calculating the area of a regular pentagon or hexagon is more complex.
# It's better to calculate the base area separately and use the general function.
def regular_polygon_area(num_sides: int, side_length: float) -> float:
    """Calculates the area of a regular polygon."""
    if num_sides < 3:
        return 0.0
    # Apothem formula: a = s / (2 * tan(180/n))
    # Area formula: A = (n * s * a) / 2
    apothem = side_length / (2 * math.tan(math.pi / num_sides))
    return (num_sides * side_length * apothem) / 2

# Example Usage:
print(f"Volume of square pyramid (side=5, height=10): {square_pyramid_volume(5, 10):.2f}")

# Example for a pentagonal pyramid
pentagon_base_area = regular_polygon_area(num_sides=5, side_length=6)
pentagonal_pyramid_vol = pyramid_volume(pentagon_base_area, height=10)
print(f"Volume of pentagonal pyramid (side=6, height=10): {pentagonal_pyramid_vol:.2f}")



a = [10,20,30,20,10,50,60,40,80,50,40]
dup_items = set()
uniq_items = []
for x in a:
    if x not in dup_items:
        uniq_items.append(x)
        dup_items.add(x)
print(dup_items) # This prints the set of duplicates, not the unique list



from typing import List, Any

def remove_duplicates_preserve_order(items: List[Any]) -> List[Any]:
    """
    Removes duplicate elements from a list while preserving the original order.

    Args:
        items: The list with potential duplicates.

    Returns:
        A new list with duplicates removed, order preserved.
    """
    # dict.fromkeys() creates a dictionary with unique keys from the list.
    # Converting it back to a list preserves the insertion order (since Python 3.7+).
    return list(dict.fromkeys(items))

def remove_duplicates_any_order(items: List[Any]) -> List[Any]:
    """
    Removes duplicate elements from a list. Order is not guaranteed.

    Args:
        items: The list with potential duplicates.

    Returns:
        A new list with duplicates removed.
    """
    return list(set(items))
    
# Example Usage:
my_list = [10, 20, 30, 20, 10, 50, 60, 40, 80, 50, 40]
unique_ordered_list = remove_duplicates_preserve_order(my_list)
print(f"Original list: {my_list}")
print(f"Unique (order preserved): {unique_ordered_list}")


def second_smallest(numbers):
    a1, a2 = float('inf'), float('inf')
    for x in numbers:
        if x <= a1:
            a1, a2 = x, a1
        elif x < a2:
            a2 = x
    return a2


from typing import List, Union

Numeric = Union[int, float]

def find_second_largest(numbers: List[Numeric]) -> Union[Numeric, None]:
    """
    Finds the second largest number in a list.

    Args:
        numbers: A list of numbers.

    Returns:
        The second largest number, or None if there isn't one.
    """
    # Remove duplicates and sort in descending order
    unique_sorted = sorted(list(set(numbers)), reverse=True)
    
    # The second largest is at index 1 if the list has at least 2 unique elements
    if len(unique_sorted) >= 2:
        return unique_sorted[1]
    return None

def find_second_smallest(numbers: List[Numeric]) -> Union[Numeric, None]:
    """
    Finds the second smallest number in a list.

    Args:
        numbers: A list of numbers.

    Returns:
        The second smallest number, or None if there isn't one.
    """
    # Remove duplicates and sort in ascending order
    unique_sorted = sorted(list(set(numbers)))

    # The second smallest is at index 1 if the list has at least 2 unique elements
    if len(unique_sorted) >= 2:
        return unique_sorted[1]
    return None

# Example Usage:
nums = [1, 2, -8, -2, 0, 8, 8]
print(f"Original numbers: {nums}")
print(f"Second smallest: {find_second_smallest(nums)}") # Should be -2
print(f"Second largest: {find_second_largest(nums)}")   # Should be 2
