import math

def factorial_iterative(n: int) -> int:
    """
    Calculates the factorial of a non-negative integer using an iterative approach.

    Args:
        n: The non-negative integer.

    Returns:
        The factorial of n.
        
    Raises:
        ValueError: If n is a negative number.
    """
    if n < 0:
        raise ValueError("Factorial does not exist for negative numbers.")
    if n == 0:
        return 1
    
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result

def factorial_recursive(n: int) -> int:
    """
    Calculates the factorial of a non-negative integer using a recursive approach.

    Args:
        n: The non-negative integer.

    Returns:
        The factorial of n.

    Raises:
        ValueError: If n is a negative number.
    """
    if n < 0:
        raise ValueError("Factorial does not exist for negative numbers.")
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial_recursive(n - 1)

# --- Main execution block ---
if __name__ == "__main__":
    num = 7
    print(f"--- Calculating factorial for {num} ---")

    # Using the iterative function
    try:
        fact_iter = factorial_iterative(num)
        print(f"Iterative approach: The factorial of {num} is {fact_iter}")
    except ValueError as e:
        print(f"Error: {e}")

    # Using the recursive function
    try:
        fact_recur = factorial_recursive(num)
        print(f"Recursive approach: The factorial of {num} is {fact_recur}")
    except ValueError as e:
        print(f"Error: {e}")

    # Using the built-in math library (the best way for real code)
    print(f"Built-in math.factorial: The factorial of {num} is {math.factorial(num)}")



def display_multiplication_table(number: int, limit: int = 10):
    """
    Displays the multiplication table for a given number up to a certain limit.

    Args:
        number: The number for which to display the table.
        limit: The upper limit for the multiplication (default is 10).
    """
    print(f"--- Multiplication Table for {number} ---")
    for i in range(1, limit + 1):
        product = number * i
        print(f"{number} x {i} = {product}")

# --- Main execution block ---
if __name__ == "__main__":
    display_multiplication_table(12)
    print("\n") # Add a newline for separation
    display_multiplication_table(9, limit=15)




def sum_natural_numbers_iterative(n: int) -> int:
    """
    Calculates the sum of natural numbers up to n using a loop.

    Args:
        n: A positive integer.

    Returns:
        The sum of natural numbers from 1 to n.
        
    Raises:
        ValueError: If n is not a positive integer.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative number.")
    
    total_sum = 0
    current_num = n
    while current_num > 0:
        total_sum += current_num
        current_num -= 1
    return total_sum

def sum_natural_numbers_recursive(n: int) -> int:
    """
    Calculates the sum of natural numbers up to n using recursion.

    Args:
        n: A positive integer.
    
    Returns:
        The sum of natural numbers from 1 to n.
    
    Raises:
        ValueError: If n is not a positive integer.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative number.")
    if n <= 1:
        return n
    else:
        return n + sum_natural_numbers_recursive(n - 1)

def sum_natural_numbers_formula(n: int) -> int:
    """
    Calculates the sum of natural numbers using the mathematical formula.
    This is the most efficient method.

    Args:
        n: A positive integer.
    
    Returns:
        The sum of natural numbers from 1 to n.
        
    Raises:
        ValueError: If n is not a positive integer.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative number.")
    return n * (n + 1) // 2

# --- Main execution block ---
if __name__ == "__main__":
    num_to_sum = 16
    print(f"--- Summing natural numbers up to {num_to_sum} ---")
    
    print(f"Iterative result: {sum_natural_numbers_iterative(num_to_sum)}")
    print(f"Recursive result: {sum_natural_numbers_recursive(num_to_sum)}")
    print(f"Formula result (most efficient): {sum_natural_numbers_formula(num_to_sum)}")
    # Pythonic way using sum() and range()
    print(f"Pythonic sum(range()) result: {sum(range(1, num_to_sum + 1))}")



import operator

def add(x, y):
    return x + y

def subtract(x, y):
    return x - y

def multiply(x, y):
    return x * y

def divide(x, y):
    if y == 0:
        raise ValueError("Cannot divide by zero.")
    return x / y

def run_calculator():
    """Runs an interactive command-line calculator."""
    
    # Using a dictionary to map choices to functions is cleaner than if/elif
    operations = {
        '1': ("Add", add),
        '2': ("Subtract", subtract),
        '3': ("Multiply", multiply),
        '4': ("Divide", divide),
    }

    print("Select operation:")
    for key, (name, _) in operations.items():
        print(f"{key}. {name}")

    while True:
        choice = input("Enter choice (1/2/3/4): ")
        
        if choice in operations:
            try:
                num1 = float(input("Enter first number: "))
                num2 = float(input("Enter second number: "))
                
                operation_name, operation_func = operations[choice]
                result = operation_func(num1, num2)
                
                # Determine the symbol for display
                op_symbol = {'Add': '+', 'Subtract': '-', 'Multiply': '*', 'Divide': '/'}[operation_name]

                print(f"Result: {num1} {op_symbol} {num2} = {result}")

            except ValueError as e:
                print(f"Invalid input: {e}. Please enter numbers.")
            except Exception as e:
                print(f"An error occurred: {e}")
            
            # Ask if user wants to perform another calculation
            next_calculation = input("Do you want to perform another calculation? (yes/no): ")
            if next_calculation.lower() != 'yes':
                print("Exiting calculator.")
                break
        else:
            print("Invalid input. Please enter a number between 1 and 4.")

# --- Main execution block ---
if __name__ == "__main__":
    run_calculator()
