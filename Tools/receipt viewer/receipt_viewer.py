
# 
# William Wadsworth
# 2.20.2025
# 
# This program allows the user to view and add receipts to an encrypted file.
# 

from typing import Dict, Union
from os import urandom, getenv, path
from dotenv import load_dotenv, set_key
from base64 import urlsafe_b64encode, b64decode, b64encode
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.fernet import Fernet, InvalidToken
from getpass import getpass

WORKSPACE_ROOT = "../"
ENV_FILE = path.join(WORKSPACE_ROOT, ".env")
ENCRYPTION_MARKER: bytes = b"ENCRYPTED::"

FILE_NAME: str = "receipts.csv"
FILE_HEADER = {
    "establishment": str,
    "total charge": float,
    "tax": float,
    "discount": float,
    "savings": float,
    "date": str,
    "payment method": str,
    "address": str,
}

PASSWORD_ENV_VAR = "RECEIPT_PASSWD"
SALT_ENV_VAR = "RECEIPT_SALT"

# --- FILE ENCRYPTION ---------------------------

class Encryptor:
    """class def."""
    
    def __init__(self):
        self.key = Encryptor.derive_key()
        self.fernet = Fernet(self.key)
        self.password_attempts: int = 3
    
    # pre-condition: 
    # post-condition: 
    @staticmethod
    def derive_key() -> bytes:
        """function def."""
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32, # key length (32 bytes for AES)
            salt=Encryptor.get_salt(),
            iterations=100000 # the larger, the harder it is to brute force
        )
        
        return urlsafe_b64encode(kdf.derive(Encryptor.get_password().encode()))

    # pre-condition: none
    # post-condition: returns the current password if a value for it exists in the environment,
    #                 otherwise the user is prompted to create one
    @staticmethod
    def get_password() -> str:
        """Fetches the password set in the environment."""
        
        # update password env
        #password = environ.get(PASSWORD_ENV_VAR)
        password = getenv(PASSWORD_ENV_VAR)
        
        if not password:
            #raise ValueError(f"Error: Environment variable '{PASSWORD_ENV_VAR}' is not set.")
            print(f"Error: Environment variable '{PASSWORD_ENV_VAR}' is not set.")
            
            # securely setup password
            password = getpass("Enter password: ")
            #environ[PASSWORD_ENV_VAR] = password 
            set_key(ENV_FILE, PASSWORD_ENV_VAR, password) # update env
        
        return password

    # pre-condition: none
    # post-condition: returns a salt if a value for it exists in the environment, otherwise a new
    #                 one is generated and the environment variable is updated
    @staticmethod
    def get_salt() -> bytes:
        """Generates a salt to be used for the encryption key."""
        
        #salt = environ.get(SALT_ENV_VAR)
        salt = getenv(SALT_ENV_VAR)
        
        if salt:
            return b64decode(salt) # convert back to bytes
        else:
            # generate a new salt if not found
            new_salt = urandom(16)
            #environ[SALT_ENV_VAR] = b64encode(new_salt).decode() 
            set_key(ENV_FILE, SALT_ENV_VAR, b64encode(new_salt).decode()) # update env
            
            return new_salt

    # pre-condition: the specified file must exist in the current directory
    # post-condition: if the file is not encrypted, its contents are encrypted and written back to
    #                 the file (overwriting), otherwise the encryption is skipped 
    def encrypt_file(self) -> None:
        """Encrypts a file's contents."""
        
        # redundant, should not be hit
        if self.is_encrypted():
            print(f"File '{FILE_NAME}' already encrypted. Skipping encryption.")
            return

        # read original file contents
        with open(FILE_NAME, "rb") as file:
            file_data = file.read()
        
        encrypted_data = self.fernet.encrypt(ENCRYPTION_MARKER + file_data)
    
        # re-write the encrypted data
        with open(FILE_NAME, "wb") as file:
            file.write(encrypted_data)

        print(f"\n{FILE_NAME} re-encrypted.")

    # pre-condition: the specified file must exist in the current directory
    # post-condition: if the file is encrypted, its contents are decrypted and written back to the
    #                 file (overwriting), otherwise the decryption is skipped 
    def decrypt_file(self) -> None:
        """Decrypts a file's contents."""
        
        # redundant, should not be hit?
        if not self.is_encrypted():
            print(f"File '{FILE_NAME}' is not encrypted. Skipping decryption.")
            return
        
        # read original file contents
        with open(FILE_NAME, "rb") as file:
            encrypted_data = file.read()
        
        decrypted_data = self.fernet.decrypt(encrypted_data)
        # remove encryption marker before writing back
        if decrypted_data.startswith(ENCRYPTION_MARKER):
            decrypted_data = decrypted_data[len(ENCRYPTION_MARKER):]

        # save decrypted data
        with open(FILE_NAME, "wb") as file:
            file.write(decrypted_data)
        
        print(f"{FILE_NAME} decrypted and ready to use.")

    # pre-condition: the specified file must exist in the current directory
    # post-condition: True is returned if the specified encryption marker precedes any data after
    #                 decrypting, otherwise (or if the decryption fails) False
    def is_encrypted(self) -> bool:
        """Check if a file is encrypted by looking for the marker."""
        
        try:
            with open(FILE_NAME, "rb") as file:
                encrypted_marker = file.read() # just check for encrypted marker

            # decrypt the marker
            decrypted_marker = self.fernet.decrypt(encrypted_marker)
            # todo: with this we are possibly decrypting twice 

            # check if it starts with the encrypted marker
            return decrypted_marker.startswith(ENCRYPTION_MARKER)

        except (InvalidToken, ValueError):
            return False # assume it's not encrypted

    # pre-condition: the password environment variable must be set up
    # post-condition: the password environment variable and class object are set to the input
    #                 password IF the user correctly inputs their current password AND the two new
    #                 password entries match
    def change_password(self) -> None:
        """Update the environment and class object's set password."""
        
        # todo: update to be like get_salt
        
        # securely setup new password
        old_password = getpass("Current password: ")
        old_password_env = getenv(PASSWORD_ENV_VAR)
        # check that the input password matches the current (old) one
        while old_password != old_password_env and self.password_attempts > 0:
            old_password = getpass(f"Incorrect. {self.password_attempts} tries remaining: ")
            self.password_attempts -= 1
        # if the user fails, do not continue
        if self.password_attempts == 0:
            print("Error: too many incorrect password attempts")
            return
        
        new_password = getpass("New password: ")
        new_password_retype = getpass("Re-type new password: ")
        
        # if the passwords match
        if new_password == new_password_retype:
            # update env and class vars
            set_key(ENV_FILE, PASSWORD_ENV_VAR, new_password)
            self.key = Encryptor.derive_key()
            self.fernet = Fernet(self.key)
            self.password_attempts = 3
            
            # if the file is already encrypted, we need to re-encrypt it
            if self.is_encrypted:
                self.encrypt_file()
        else:
            print("Error: password mismatch. New password not set")
            return
    
# --- ENTRY MANAGEMENT --------------------------

# pre-condition: 
# post-condition: 
def validate_entry(entry: Dict[str, Union[str, float]]) -> bool:
    """Validate that all fields have the correct type."""

    for key, expected_type in FILE_HEADER.items():
        if key not in entry:
            print(f"Missing field: {key}")
            return False
        if not isinstance(entry[key], expected_type):
            print(f"Invalid type for {key}. Expected {expected_type}, got {type(entry[key])}")
            return False
    
    return True

# pre-condition: 
# post-condition: 
def add_entry(
    new_entry: Dict[str, Union[str, float]],
    filename: str
) -> None:
    """Adds a line to the specified file."""
    
    if validate_entry(new_entry):
        try:
            with open(filename,'a') as file:
                file.write(','.join(new_entry) + '\n')
        except IOError as e:
            print(f"Error writing to file: {e}")
    else:
        print(f"Error, entry contains an invalid type: {new_entry}")

# pre-condition: 
# post-condition: 
def read_entries() -> None:
    """Prints each line in the specified file."""
    
    try:
        with open(FILE_NAME,'r') as file:
            for line in file:
                print(line.strip())
    except FileNotFoundError:
        print(f"Error: file '{FILE_NAME}' does not exist.")

# ---  -------------

# pre-condition: 
# post-condition: 
def create_file(encrypter: Encryptor):
    """function def."""
    
    if load_dotenv():
        # does the file exist?
        if path.isfile(FILE_NAME):
            if encrypter.is_encrypted(): encrypter.decrypt_file()
            else: return
        else:
            # create the file and add header
            try:
                with open(FILE_NAME, 'w') as file:
                    file.write(FILE_HEADER+'\n')
            except Exception as e:
                raise Exception(f"Error opening file '{FILE_NAME}': {e}.")
            
            # todo: prompt for password change
    else: # todo: is this even necessary? or change to instead add required envs
        raise Exception("No environment variables. Please add required environment variables")


def main():
    encrypter = Encryptor()
    create_file(encrypter)
    
    try:
        while True:
            choice = int(input("\n\n1: add, 2: read: "))
            if choice == 1:
                new_items = []
                for item in FILE_HEADER:
                    new_items.append(input(f"{item}: "))
                
                add_entry(new_items,FILE_NAME)
            else: read_entries()
    except KeyboardInterrupt:
        encrypter.encrypt_file()
    
if __name__ == "__main__":
    main()