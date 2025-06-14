# test.py
import valipak
valipak.is_in_range
# Test the string validator
email1 = "test@example.com"
print(f"Is '{email1}' a valid email? {valipak.is_valid_email(email1)}")

# Test the numeric validator
num1 = 50
print(f"Is {num1} between 1 and 100? {valipak.is_in_range(num1, 1, 100)}")