
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from base64 import urlsafe_b64encode, urlsafe_b64decode

#Encrypting data using deterministic methodology
def encrypt_deterministic(key, plaintext):    
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    encryptor = cipher.encryptor()
    data_to_encrypt = str(plaintext).encode()
    padded_plaintext = data_to_encrypt.ljust(16) # Pad to block size (16 bytes for AES)
    ciphertext = encryptor.update(padded_plaintext) + encryptor.finalize()    
    return urlsafe_b64encode(ciphertext)
#1 - one encryption
#2  - one encryption

#Decrypting data using deterministic methodology
def decrypt_deterministic(key, ciphertext):
    cipher = Cipher(algorithms.AES(key), modes.ECB(), backend=default_backend())
    decryptor = cipher.decryptor()
    decrypted = decryptor.update(urlsafe_b64decode(ciphertext)) + decryptor.finalize()
    return decrypted.rstrip(b'\0') # Remove padding

def getKey():
    # Generate a proper AES key using PBKDF2    
    password = b'password'
    salt = b'\xeb\xd0\xd7W\xe8\x15w7\xcf0\x1f]\xe2x\xcd\xc8' #Salt and password needs to be kept save
    kdf = PBKDF2HMAC(
    algorithm=hashes.SHA256(),
    iterations=100000,
    salt=salt,length=32, backend=default_backend() ) # Key size for AES-256
    key = kdf.derive(password)
    return key


