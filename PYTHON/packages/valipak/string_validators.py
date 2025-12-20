# valipak/string_validators.py
import re

def is_valid_email(email: str) -> bool:
    """Checks if the provided string is a validly formatted email address."""
    if not isinstance(email, str):
        return False
    pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    return re.match(pattern, email) is not None