
# 
# William Wadsworth
# 03.09.2025
# 

from dotenv import set_key
from getpass import getpass
from os import getenv, path

WORKSPACE_ROOT = "../"
ENV_FILE = path.join(WORKSPACE_ROOT, ".env")
PASSWORD_ENV_VAR = "DAVID_DATABASE_PASSWORD"

class PasswordManager():
    """class def."""
    
    def __init__(self):
        self.__password = getenv(PASSWORD_ENV_VAR)
        self.password_attempts: int = 3
        
        if not self.__password:
            print(f"Error: Environment variable '{PASSWORD_ENV_VAR}' is not set.")
            self.prompt_password()
    
    # pre-condition: 
    # post-condition: 
    def prompt_password(self) -> None:
        """function def."""
        
        # securely setup password
        password = getpass("Enter password: ")
        self.__password = password
        #environ[PASSWORD_ENV_VAR] = password 
        set_key(ENV_FILE, PASSWORD_ENV_VAR, password) # update env
    
    # pre-condition: 
    # post-condition: 
    def get_password(self) -> str: 
        """function def."""
        
        return self.__password
    
    # pre-condition: the password environment variable must be set up
    # post-condition: the password environment variable and class object are set to the input
    #                 password IF the user correctly inputs their current password AND the two new
    #                 password entries match
    def change_password(self) -> None:
        """Update the environment and class object's set password."""
        
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
            self.__password = new_password
            self.password_attempts = 3
        else:
            print("Error: password mismatch. New password not set")
            return

