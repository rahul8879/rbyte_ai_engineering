# valipak/numeric_validators.py
from typing import Union

def is_in_range(number: Union[int, float], min_val: int, max_val: int) -> bool:
    """Checks if a number is within a specified range (inclusive)."""
    if not isinstance(number, (int, float)):
        return False
    return min_val <= number <= max_val