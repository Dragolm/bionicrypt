import hashlib

# THE_KEY = 53599889662614725004850542191481889509

def keyGiver(string: str):
    hash_bytes = hashlib.md5(string.encode()).digest()
    hash_int = int.from_bytes(hash_bytes, 'big')
    print(hash_int)
    return hash_int