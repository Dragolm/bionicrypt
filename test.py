import hashlib

def hash_string(s):
    # Create MD5 hash of the string
    hash_bytes = hashlib.md5(s.encode()).digest()
    # Convert first 4 bytes to int and mod by 1001
    hash_int = int.from_bytes(hash_bytes, 'big')
    return hash_int

# Example
print(hash_string("ash"))  # Always same result
print(hash_string("worldavnlsvrvs"))