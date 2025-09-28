from typing import List

def heapify(arr: List[int], heap_size: int, root_index: int) -> None:
    """
    Ensure the subtree rooted at root_index is a max heap.

    Args:
        arr: The list of numbers to be sorted.
        heap_size: The size of the heap to consider.
        root_index: The root index of the subtree.
    """
    largest_index = root_index
    left_child_index = 2 * root_index + 1
    right_child_index = 2 * root_index + 2

    # Check if left child exists and is greater than the root
    if left_child_index < heap_size and arr[left_child_index] > arr[largest_index]:
        largest_index = left_child_index

    # Check if right child exists and is greater than the largest so far
    if right_child_index < heap_size and arr[right_child_index] > arr[largest_index]:
        largest_index = right_child_index

    # If the largest element is not the root, swap them and heapify the affected subtree
    if largest_index != root_index:
        arr[root_index], arr[largest_index] = arr[largest_index], arr[root_index]
        heapify(arr, heap_size, largest_index)

def heap_sort(arr: List[int]) -> None:
    """
    Sorts a list of numbers in place using the Heap Sort algorithm.

    Args:
        arr: The list of numbers to be sorted.
    """
    n = len(arr)

    # 1. Build a max heap from the list.
    # The last parent node is at index (n // 2) - 1. We iterate backwards from there.
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 2. Extract elements one by one from the heap.
    for i in range(n - 1, 0, -1):
        # Move the current root (largest element) to the end of the list
        arr[i], arr[0] = arr[0], arr[i]
        # Call heapify on the reduced heap
        heapify(arr, i, 0)



from typing import List

def _partition(arr: List[int], low: int, high: int) -> int:
    """
    Partitions the array around a pivot element.

    Elements smaller than the pivot are moved to its left, and larger elements
    to its right. The pivot is the last element of the sub-array.

    Args:
        arr: The list of numbers.
        low: The starting index of the sub-array.
        high: The ending index of the sub-array.

    Returns:
        The index where the pivot element is finally placed.
    """
    pivot = arr[high]
    # index of smaller element
    i = low - 1

    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]

    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

def _quick_sort_recursive(arr: List[int], low: int, high: int) -> None:
    """
    The recursive helper function that implements the Quick Sort algorithm.

    Args:
        arr: The list of numbers.
        low: The starting index for sorting.
        high: The ending index for sorting.
    """
    if low < high:
        partition_index = _partition(arr, low, high)
        _quick_sort_recursive(arr, low, partition_index - 1)
        _quick_sort_recursive(arr, partition_index + 1, high)

def quick_sort(arr: List[int]) -> None:
    """
    Sorts a list of numbers in place using the Quick Sort algorithm.

    Args:
        arr: The list of numbers to be sorted.
    """
    _quick_sort_recursive(arr, 0, len(arr) - 1)




def has_consecutive_zeros_in_base_k(n: int, k: int) -> bool:
    """
    Checks if a number has consecutive zeros in its base-k representation.

    Args:
        n: The decimal number to check.
        k: The base to convert to (must be >= 2).

    Returns:
        True if the base-k representation of n contains consecutive zeros,
        False otherwise.
    """
    if k < 2:
        raise ValueError("Base k must be 2 or greater.")
    if n == 0:
        return False # Single '0' does not have consecutive zeros

    base_k_digits = []
    temp_n = n
    while temp_n > 0:
        base_k_digits.append(str(temp_n % k))
        temp_n //= k

    base_k_string = "".join(reversed(base_k_digits))
    return "00" in base_k_string

# Example usage
if __name__ == "__main__":
    num_to_check = 20
    base = 2 # 20 in base 2 is 10100, which has consecutive zeros
    if has_consecutive_zeros_in_base_k(num_to_check, base):
        print(f"Yes, {num_to_check} has consecutive zeros in base {base}.")
    else:
        print(f"No, {num_to_check} does not have consecutive zeros in base {base}.")




def get_min_sum_of_factors(n: int) -> int:
    """
    Calculates the sum of the prime factors of a number.
    This is equivalent to finding the minimum sum of numbers that multiply to n.

    Args:
        n: The number to factorize.

    Returns:
        The sum of its prime factors.
    """
    factor_sum = 0
    i = 2
    
    # Process factors of 2
    while n % i == 0:
        factor_sum += i
        n //= i

    # Process odd factors
    i = 3
    while i * i <= n:
        while n % i == 0:
            factor_sum += i
            n //= i
        i += 2
        
    # If n is still a prime number greater than 2
    if n > 1:
        factor_sum += n

    return factor_sum




from typing import Any, Optional, List

class CircularQueue:
    """A circular queue implementation with a fixed size."""

    def __init__(self, limit: int = 10):
        """
        Initializes the circular queue.

        Args:
            limit: The maximum size of the queue.
        """
        self.limit = limit
        self.queue: List[Optional[Any]] = [None] * limit
        self.front = -1
        self.rear = -1

    def __str__(self) -> str:
        """Returns a string representation of the queue's contents."""
        if self.is_empty():
            return "[]"

        items = []
        if self.rear >= self.front:
            for i in range(self.front, self.rear + 1):
                items.append(str(self.queue[i]))
        else:
            for i in range(self.front, self.limit):
                items.append(str(self.queue[i]))
            for i in range(0, self.rear + 1):
                items.append(str(self.queue[i]))
        return f"[{', '.join(items)}]"

    def is_empty(self) -> bool:
        """Checks if the queue is empty."""
        return self.front == -1

    def is_full(self) -> bool:
        """Checks if the queue is full."""
        return (self.rear + 1) % self.limit == self.front

    def enqueue(self, data: Any) -> None:
        """
        Adds an element to the rear of the queue.

        Args:
            data: The data to be added.
        
        Raises:
            IndexError: If the queue is full.
        """
        if self.is_full():
            raise IndexError("Queue is full.")
        elif self.is_empty():
            self.front = 0
            self.rear = 0
        else:
            self.rear = (self.rear + 1) % self.limit
        self.queue[self.rear] = data

    def dequeue(self) -> Any:
        """
        Removes and returns the element from the front of the queue.

        Returns:
            The data from the front of the queue.
        
        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError("Queue is empty.")
        
        data = self.queue[self.front]
        if self.front == self.rear:
            # Queue becomes empty after this dequeue
            self.front = -1
            self.rear = -1
        else:
            self.front = (self.front + 1) % self.limit
        return data




import collections
from typing import Any

class Deque:
    """
    A double-ended queue (deque) implementation with a fixed size limit,
    wrapping the highly optimized collections.deque.
    """
    def __init__(self, limit: int = 10):
        """
        Initializes the deque.

        Args:
            limit: The maximum size of the deque.
        """
        self._deque = collections.deque(maxlen=limit)
        self.limit = limit

    def __str__(self) -> str:
        """Returns a string representation of the deque."""
        return f"[{', '.join(str(item) for item in self._deque)}]"

    def is_empty(self) -> bool:
        """Checks if the deque is empty."""
        return len(self._deque) == 0

    def is_full(self) -> bool:
        """Checks if the deque is full."""
        return len(self._deque) == self.limit

    def insert_rear(self, data: Any) -> None:
        """
        Adds an element to the rear (right side) of the deque.

        Args:
            data: The element to add.
        
        Raises:
            IndexError: If the deque is full.
        """
        if self.is_full():
            raise IndexError("Deque is full.")
        self._deque.append(data)

    def insert_front(self, data: Any) -> None:
        """
        Adds an element to the front (left side) of the deque.

        Args:
            data: The element to add.

        Raises:
            IndexError: If the deque is full.
        """
        if self.is_full():
            raise IndexError("Deque is full.")
        self._deque.appendleft(data)

    def delete_rear(self) -> Any:
        """
        Removes and returns the element from the rear (right side).

        Returns:
            The element from the rear.

        Raises:
            IndexError: If the deque is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot delete from an empty deque.")
        return self._deque.pop()

    def delete_front(self) -> Any:
        """
        Removes and returns the element from the front (left side).

        Returns:
            The element from the front.
        
        Raises:
            IndexError: If the deque is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot delete from an empty deque.")
        return self._deque.popleft()




import heapq
from typing import List, Any, Union

# Python's heapq is a min-heap. To simulate a max-heap, we can store negated values.
# This implementation assumes we are storing numbers. For complex objects,
# a custom wrapper class or tuple `(-priority, item)` would be used.
Numeric = Union[int, float]

class PriorityQueue:
    """
    A max-priority queue implementation using Python's heapq (min-heap) module.
    """

    def __init__(self):
        """Initializes an empty priority queue."""
        self._queue: List[Numeric] = []

    def __str__(self) -> str:
        """Returns a string representation of the queue's contents (in max-order)."""
        # Create a sorted list of the positive values for printing
        sorted_items = sorted([-item for item in self._queue], reverse=True)
        return f"[{', '.join(str(item) for item in sorted_items)}]"

    def is_empty(self) -> bool:
        """Checks if the priority queue is empty."""
        return len(self._queue) == 0

    def insert(self, data: Numeric) -> None:
        """
        Adds an item to the priority queue.

        Args:
            data: A numeric item to add to the queue.
        """
        # Negate the item to simulate a max-heap using a min-heap
        heapq.heappush(self._queue, -data)

    def delete(self) -> Numeric:
        """
        Removes and returns the item with the highest priority (the largest number).

        Returns:
            The item with the highest priority.

        Raises:
            IndexError: If the queue is empty.
        """
        if self.is_empty():
            raise IndexError("Cannot delete from an empty priority queue.")
        # heappop returns the smallest item, which is the negative of our largest number
        return -heapq.heappop(self._queue)
