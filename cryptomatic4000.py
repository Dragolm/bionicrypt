from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
import os
from facialVaultifier import verifier, capture_image, enroller
import cv2
import evaluator

userName = "john cena"

class PasswordFileEncryptor:
    def __init__(self, password):
        self.password = password.encode()
    
    def _derive_key(self, salt):
        # Derive encryption key from password
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=480000,
        )
        return kdf.derive(self.password)
    
    def encrypt_file(self, input_path, output_path):
        """Encrypt file with password"""
        # Generate random salt
        salt = os.urandom(16)
        key1 = self._derive_key(salt)
        key2 = self._derive_key(salt)
        print("key2 = " + str(key2.hex()))
        print("key1 = " + str(key1.hex()))
        
        # Read file
        with open(input_path, 'rb') as f:
            plaintext = f.read()
        
        # Encrypt
        cipher = AESGCM(key1)
        nonce = os.urandom(12)
        ciphertext = cipher.encrypt(nonce, plaintext, None)
        
        # Write: salt + nonce + ciphertext
        with open(output_path, 'wb') as f:
            f.write(salt + nonce + ciphertext)
        
        print(f"✓ Encrypted: {input_path}")
    
    def decrypt_file(self, input_path, output_path):
        """Decrypt file with password"""
        # Read encrypted file
        with open(input_path, 'rb') as f:
            encrypted_data = f.read()
        from cryptography.hazmat.primitives import hashes
        # Extract components
        salt = encrypted_data[:16]
        nonce = encrypted_data[16:28]
        ciphertext = encrypted_data[28:]
        
        # Derive key from password
        key = self._derive_key(salt)
        
        # Decrypt
        cipher = AESGCM(key)
        plaintext = cipher.decrypt(nonce, ciphertext, None)
        
        # Write decrypted file
        with open(output_path, 'wb') as f:
            f.write(plaintext)
        
        print(f"✓ Decrypted: {input_path}")

mode = input("Enter mode (enroll/verify): ").strip().lower()
# For test case can be removed in production
if input("Is this a test?(y/n)").strip().lower()=="y":
    image = cv2.imread("athul.png")
else:
    image = capture_image()
THE_KEY = evaluator.keyGiver(userName)

if mode in ["enroll", "e"]:
    enroller(image, THE_KEY)
else:
    password = str(verifier(image, THE_KEY))

    # Usage with password
    # password = "MySecurePassword123!"
    encryptor = PasswordFileEncryptor(password)

    # Encrypt/home/the-forge/Desktop/trashable/pybrakeTrialShit/vacation.png
    encryptor.encrypt_file('vacation.png', 'vacation.png.enc')
    encryptor.encrypt_file('family_video.mp4', 'family_video.mp4.enc')

    # Decrypt (need same password)
    encryptor.decrypt_file('vacation.png.enc', 'vacation_restored.png')
    encryptor.decrypt_file('family_video.mp4.enc', 'family_video_restored.mp4')