# -*- coding: utf-8 -*-
"""
A comprehensive collection of Python code examples, refactored for clarity,
readability, and adherence to best practices.

This script serves as a learning resource and a demonstration of standard Python
conventions, covering topics from basic functions to advanced concepts like
closures, decorators, classes, iterators, and common language pitfalls.
"""

# --- 1. IMPORTS ---
import datetime
import inspect
import random
import time
from collections import namedtuple
from copy import deepcopy
from time import perf_counter
from timeit import timeit
from typing import (Any, Callable, Dict, Iterable, List, Optional, Sequence,
                    Tuple, Union)

# Third-party libraries (require installation: pip install pytz)
try:
    import pytz
except ImportError:
    print("Warning: 'pytz' is not installed. Timezone-related functions may fail.")
    pytz = None


# --- 2. DATE AND TIME FUNCTIONS ---

def get_utc_now() -> datetime.datetime:
    """Returns the current time in UTC."""
    return datetime.datetime.utcnow()


def get_india_now() -> Optional[datetime.datetime]:
    """Returns the current time in the 'Asia/Kolkata' timezone."""
    if not pytz:
        return None
    india_tz = pytz.timezone('Asia/Kolkata')
    return datetime.datetime.now(india_tz)


def get_yesterday_and_tomorrow() -> Tuple[datetime.datetime, datetime.datetime]:
    """Returns the datetime objects for yesterday and tomorrow."""
    now = datetime.datetime.now()
    yesterday = now - datetime.timedelta(days=1)
    tomorrow = now + datetime.timedelta(days=1)
    return yesterday, tomorrow


# --- 3. FUNCTIONAL PROGRAMMING, DECORATORS, AND CLOSURES ---

def universal_function(*args: Any, **kwargs: Any) -> None:
    """Demonstrates a function that accepts any number of arguments."""
    print("Positional arguments (args):", args)
    print("Keyword arguments (kwargs):", kwargs)


def log_message(
    message: str,
    timestamp: Optional[datetime.datetime] = None
) -> None:
    """
    Logs a message with a timestamp.

    Note: Avoids the mutable default argument pitfall by setting the
    timestamp inside the function if it's not provided.
    """
    if timestamp is None:
        timestamp = datetime.datetime.utcnow()
    print(f'Message at {timestamp}: {message}')


def factorial_recursive(n: int) -> int:
    """
    Calculates the factorial of a number using recursion.

    Args:
        n: A non-negative integer.

    Returns:
        The factorial of n.
    """
    if n < 1:
        return 1
    return n * factorial_recursive(n - 1)


def factorial_terse(n: int) -> int:
    """Calculates factorial using a one-line ternary operator."""
    return 1 if n < 2 else n * factorial_terse(n - 1)


def apply_function(x: Any, func: Callable[[Any], Any]) -> Any:
    """
    A higher-order function that applies a given function to a value.

    Args:
        x: The input value.
        func: The function to apply to x.

    Returns:
        The result of func(x).
    """
    return func(x)


def get_function_info(func: Callable) -> None:
    """
    Inspects an object to check if it's a function or a method.

    Args:
        func: The object to inspect.
    """
    name = getattr(func, '__name__', 'N/A')
    print(f"'{name}' is a method: {inspect.ismethod(func)}")
    print(f"'{name}' is a function: {inspect.isfunction(func)}")


def get_source_code(obj: Any) -> str:
    """
    Returns the source code of a function, class, or module as a string.
    """
    return inspect.getsource(obj)


def dot_product(vector1: Sequence[float], vector2: Sequence[float]) -> float:
    """Calculates the dot product of two numerical vectors."""
    return sum(a * b for a, b in zip(vector1, vector2))


# --- Closures and Decorators ---

def make_counter() -> Callable[[], None]:
    """
    A closure that creates a counter. Each call to the returned function
    increments and prints the count.
    """
    count = 0
    def inner() -> None:
        nonlocal count
        count += 1
        print(f'This function has been called {count} times.')
    return inner


def make_timed_logger() -> Callable[[], None]:
    """
    A closure that creates a logger with a persistent timestamp.
    """
    count = 0
    start_time = datetime.datetime.now()
    def inner() -> None:
        nonlocal count
        count += 1
        print(f'Function called {count} times. First call was at {start_time}.')
    return inner


def time_elapsed_since_creation() -> Callable[[], float]:
    """A closure that returns the time elapsed since it was created."""
    start_time = perf_counter()
    def inner() -> float:
        return perf_counter() - start_time
    return inner


def with_counter(fn: Callable) -> Callable:
    """A decorator that adds a call counter to a function."""
    count = 0
    def inner(*args, **kwargs) -> Any:
        nonlocal count
        count += 1
        print(f"'{fn.__name__}' has been called {count} times.")
        return fn(*args, **kwargs)
    return inner


def with_logging(fn: Callable) -> Callable:
    """A decorator that logs when a function is called and with what arguments."""
    def inner(*args, **kwargs) -> Any:
        timestamp = datetime.datetime.now()
        print(
            f"Calling '{fn.__name__}' at {timestamp} "
            f"with args: {args}, kwargs: {kwargs}"
        )
        return fn(*args, **kwargs)
    return inner


# --- 4. DATA STRUCTURE MANIPULATIONS ---

def string_to_list(sentence: str) -> List[str]:
    """Converts a string into a list of its characters."""
    return list(sentence)


def string_to_tuple(sentence: str) -> Tuple[str, ...]:
    """Converts a string into a tuple of its characters."""
    return tuple(sentence)


def zip_sequences(seq1: Sequence, seq2: Sequence) -> zip:
    """Zips two sequences together."""
    return zip(seq1, seq2)


def check_all_true(items: Iterable) -> bool:
    """Returns True if all items in the iterable are truthy, otherwise False."""
    return all(items)


def check_any_true(items: Iterable) -> bool:
    """Returns True if any item in the iterable is truthy, otherwise False."""
    return any(items)


def unpack_tuple_parts(data: Tuple) -> Tuple[Any, Any, List[Any]]:
    """
    Unpacks a tuple into its first two elements and a list of the rest.
    Requires the tuple to have at least two elements.
    """
    if len(data) < 2:
        raise ValueError("Input tuple must have at least two elements.")
    first, second, *rest = data
    return first, second, rest


def sum_unlimited_args(*args: Union[int, float]) -> Union[int, float]:
    """Adds an unlimited number of numerical arguments."""
    return sum(args)


IplData = namedtuple(
    'IplData', 'match toss choice session1 session2 winner'
)
IplData.__doc__ = 'Stores data for an IPL match.'
IplData.match.__doc__ = 'Description of the match (e.g., "TeamA vs TeamB").'
IplData.toss.__doc__ = 'The team that won the toss.'
IplData.choice.__doc__ = 'The choice made after winning the toss (e.g., "bat" or "field").'
IplData.session1.__doc__ = 'Score of the first batting team.'
IplData.session2.__doc__ = 'Score of the second batting team.'
IplData.winner.__doc__ = 'The winning team.'

def create_ipl_match_record(data: Tuple) -> IplData:
    """Creates a structured IplData namedtuple from a tuple."""
    return IplData(*data)


# --- 5. CLASSES AND OBJECT-ORIENTED PROGRAMMING ---

class User:
    """A simple class demonstrating custom string representations."""
    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        """Returns the official, unambiguous string representation of the object."""
        return f"User(name='{self.name}')"

    def __str__(self) -> str:
        """Returns the informal, user-friendly string representation."""
        return f"User: {self.name}"


class ComparableValue:
    """A class that implements custom equality and less-than comparisons."""
    def __init__(self, value: int):
        self.value = value

    def _validate_other(self, other: Any) -> 'ComparableValue':
        if not isinstance(other, ComparableValue):
            raise TypeError(
                "Comparison not supported between 'ComparableValue' and "
                f"'{type(other).__name__}'"
            )
        return other

    def __eq__(self, other: Any) -> bool:
        """Checks for equality based on the 'value' attribute."""
        try:
            other_obj = self._validate_other(other)
            return self.value == other_obj.value
        except TypeError:
            return NotImplemented

    def __lt__(self, other: Any) -> bool:
        """Checks for less-than based on the 'value' attribute."""
        try:
            other_obj = self._validate_other(other)
            return self.value < other_obj.value
        except TypeError:
            return NotImplemented


class CallableClass:
    """An example of a class whose instances are callable."""
    def __call__(self, x: int, y: int) -> int:
        """Makes instances of this class behave like a function."""
        print(f"Instance called with {x} and {y}")
        return x + y


class MathUtils:
    """A utility class demonstrating a static method."""
    @staticmethod
    def add(a: float, b: float) -> float:
        """A static method to add two numbers. No instance is needed."""
        return a + b


class CustomSequence:
    """A class that behaves like a sequence (supports len() and indexing)."""
    def __init__(self, data: Sequence):
        self._data = list(data)

    def __len__(self) -> int:
        """Returns the length of the sequence."""
        return len(self._data)

    def __getitem__(self, index: int) -> Any:
        """
        Retrieves an item by index, supporting negative indexing.

        Raises:
            IndexError: If the index is out of bounds.
        """
        if isinstance(index, int):
            return self._data[index]
        raise TypeError("Index must be an integer.")


# --- 6. ITERATORS AND GENERATORS ---

def fibonacci_generator(n: int):
    """
    A generator that yields the first n Fibonacci numbers.
    """
    a, b = 0, 1
    for _ in range(n):
        yield a
        a, b = b, a + b


class RandomIntegerIterator:
    """
    An iterator that generates a specified number of random integers.
    The class itself is the iterator.
    """
    def __init__(self, count: int, min_val: int, max_val: int):
        self.count = count
        self.min_val = min_val
        self.max_val = max_val

    def __iter__(self):
        return self

    def __next__(self) -> int:
        if self.count > 0:
            self.count -= 1
            return random.randint(self.min_val, self.max_val)
        raise StopIteration


class Squares:
    """
    An iterable that generates squares up to a number `n`.
    It is both an iterable (using a generator) and a sequence (supports indexing).
    """
    def __init__(self, n: int):
        self.n = n

    def __iter__(self):
        """Returns a generator for iterating over squares."""
        for i in range(self.n):
            yield i ** 2

    def __getitem__(self, index: int) -> int:
        """Returns the square of the index directly."""
        if not isinstance(index, int):
            raise TypeError("Index must be an integer.")
        if 0 <= index < self.n:
            return index ** 2
        raise IndexError("Index out of range.")


# --- 7. DEMONSTRATIONS OF PYTHON CONCEPTS AND PITFALLS ---

def demonstrate_boolean_as_number():
    """Shows that `bool` is a subclass of `int` where True=1 and False=0."""
    print("\n--- Booleans as Numbers ---")
    print(f"True * 5 = {True * 5}")   # Output: 5
    print(f"False * 5 = {False * 5}")  # Output: 0
    print(f"isinstance(True, int) is {isinstance(True, int)}")


def demonstrate_scope():
    """Illustrates global, local, and nonlocal scopes."""
    print("\n--- Variable Scopes ---")
    val = 10  # Global variable

    def use_global_scope(n: int) -> int:
        """Accesses and uses the global `val`."""
        # 'global val' would be needed to modify it
        return val * n

    def use_nonlocal_scope():
        """Shows how 'nonlocal' modifies a variable in an enclosing scope."""
        x = 'Outer'
        def inner():
            nonlocal x
            x = 'Inner'
            print(f'  inner() scope: x = {x}')
        print(f'Before inner() call: x = {x}')
        inner()
        print(f'After inner() call: x = {x}')

    print(f"Global 'val' is {val}")
    print(f"Result of use_global_scope(5): {use_global_scope(5)}")
    use_nonlocal_scope()


def demonstrate_mutable_default_argument_pitfall():
    """
    Highlights the danger of using mutable objects as default arguments.
    """
    print("\n--- Mutable Default Argument Pitfall ---")

    # Incorrect implementation
    def append_to_list_bad(value, def_list=[]):
        def_list.append(value)
        return def_list

    # Correct implementation
    def append_to_list_good(value, def_list=None):
        if def_list is None:
            def_list = []
        def_list.append(value)
        return def_list

    list1 = append_to_list_bad(1)
    print(f"Bad - First call: {list1}")
    list2 = append_to_list_bad(2)
    print(f"Bad - Second call: {list2} (mutated the original list!)")

    list3 = append_to_list_good(1)
    print(f"Good - First call: {list3}")
    list4 = append_to_list_good(2)
    print(f"Good - Second call: {list4} (created a new list)")


def demonstrate_identity_vs_equality():
    """
    Explains the difference between `is` (identity) and `==` (equality).
    """
    print("\n--- Identity (`is`) vs. Equality (`==`) ---")
    list_a = [1, 2, 3]
    list_b = [1, 2, 3]
    list_c = list_a

    print(f"list_a == list_b: {list_a == list_b} (Values are equal)")
    print(f"list_a is list_b: {list_a is list_b} (Objects are different)")
    print(f"list_a is list_c: {list_a is list_c} (Same object)")

    # CPython optimization for small integers
    a, b = 256, 256
    c, d = 257, 257
    print(f"256 is 256: {a is b} (Cached)")
    print(f"257 is 257: {c is d} (Not cached, new objects)")


def demonstrate_shallow_vs_deep_copy():
    """Illustrates the difference between shallow and deep copies."""
    print("\n--- Shallow vs. Deep Copy ---")
    list1 = [[1, 2], [3, 4]]

    # Shallow copy
    list2 = list1.copy()
    # Deep copy
    list3 = deepcopy(list1)

    # Modify a nested object in the original list
    list1[0][0] = 99

    print(f"Original list: {list1}")
    print(f"Shallow copy:  {list2} (nested list was affected!)")
    print(f"Deep copy:     {list3} (nested list is independent)")


# --- 8. MAIN EXECUTION BLOCK ---

def main():
    """
    Main function to demonstrate the usage of the refactored code.
    """
    print("--- Running Code Demonstrations ---")

    # Date and Time
    print("\n--- Date and Time ---")
    print(f"Current UTC time: {get_utc_now()}")
    if pytz:
        print(f"Current Indian time: {get_india_now()}")
    yesterday, tomorrow = get_yesterday_and_tomorrow()
    print(f"Yesterday was: {yesterday.date()}, Tomorrow will be: {tomorrow.date()}")

    # Higher-order function
    print("\n--- Functional Concepts ---")
    result = apply_function(5, lambda x: x * x)
    print(f"Square of 5 is: {result}")

    # Decorator example
    @with_logging
    def greet(name):
        print(f"Hello, {name}!")

    greet("World")

    # Class example
    print("\n--- Classes and OOP ---")
    user1 = User("Alice")
    print(str(user1))   # User-friendly output
    print(repr(user1))  # Developer-friendly output

    # Generator example
    print("\n--- Generators ---")
    fib_numbers = list(fibonacci_generator(10))
    print(f"First 10 Fibonacci numbers: {fib_numbers}")

    # Pitfall demonstrations
    demonstrate_boolean_as_number()
    demonstrate_scope()
    demonstrate_mutable_default_argument_pitfall()
    demonstrate_identity_vs_equality()
    demonstrate_shallow_vs_deep_copy()


if __name__ == "__main__":
    main()
