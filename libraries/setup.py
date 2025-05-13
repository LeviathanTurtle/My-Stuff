
# 
# William Wadsworth
# 05.03.2025
# 
# A collection of functions to be used when constructing an environment
# 

import os
import sys
import subprocess
import structlog
from dotenv import load_dotenv, set_key

REQUIREMENTS = "requirements.txt"
ENV_FILE = ".env"

def is_virtualenv() -> bool:
    """Helper function to check if the script is running inside a virtual environment."""
    
    return (hasattr(sys, 'real_prefix') or # for older python versions
           (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)) # newer

def build_env() -> None:
    """Ensure missing environment variables are added to the .env file."""
    
    # --- SETUP ---
    logger = structlog.get_logger()

    # ensure .env file exists
    if not os.path.exists(ENV_FILE):
        logger.warning(f"'{ENV_FILE}' not found. Creating a new one.")
        with open(ENV_FILE, "w") as f:
            f.write("# Auto-generated .env file\n")
    
    if load_dotenv():
        logger.info(f"Environment in {__file__} successfully loaded")
    else: logger.error(f"Environment in {__file__} could not load")

    VARIABLES = {
        #"APP_ADDRESS" : os.getenv("APP_ADDRESS", "0.0.0.0"),
        # ...
    }
    logger.debug(f"Environment variables: {VARIABLES}")
    
    def print_variables() -> None:
        """Helper function to list all required environment variables and their values."""
        
        print()
        for var, value in VARIABLES.items():
            print(f"{var}: {value}")

    def verify_values() -> None:
        """List required environment variables and prompt for updates."""
        
        print_variables()
        choice = int(input("Enter 0 to continue, or 1 to edit one or more values: "))
        if choice == 0: return
        
        while True:
            logger.info("User is choosing to update an environment variable")
            print_variables()
            
            print("\nAt any point, enter 2 to exit.")
            key_to_edit = input("Select a value to edit: ")
            if key_to_edit == "2": return 
            new_value = input("Enter the new value: ")
            if new_value == "2": return
            
            try:
                VARIABLES[f'{key_to_edit}'] = new_value
                set_key(ENV_FILE, f'{key_to_edit}', new_value)
                logger.info(f"User has updated {VARIABLES[f'{key_to_edit}']} to {new_value}")
            except Exception as e:
                logger.exception(f"Error updating '{key_to_edit}' with value '{new_value}'", exc_info=e)


    # --- MAIN FUNCTION STUFF ---
    verify_values()
    # todo restructure all this
    existing_vars = {}
    if os.path.exists(ENV_FILE):
        with open(ENV_FILE, 'r') as file:
            for line in file:
                if "=" in line:
                    key, val = line.strip().split("=", 1)
                    existing_vars[key] = val
    logger.debug(f"Found variables in the environment: {existing_vars}")
    
    missing_vars = {k: v for k, v in VARIABLES.items() if k not in existing_vars}
    logger.debug(f"Missing variables: {missing_vars}")
    
    if missing_vars:
        with open(ENV_FILE, 'a') as file:
            for var, value in missing_vars.items():
                #file.write(f"{var}={value}\n")
                file.write(f"{var}=\"{value}\"\n")
                logger.debug(f"Added package {var} ({value}) to environment")

def install_missing_packages() -> None:
    """Install or update only missing or outdated packages from the requirements file."""
    
    logger = structlog.get_logger()
    
    if not is_virtualenv():
        logger.info("User is not in a virtual environment")
        print("[WARNING] It is recommended to run this script inside a virtual environment.")
        print("Would you like to create one now? [Yes/No]")
        response = input(": ").strip().lower()
        
        if response in ("yes", "y"):
            logger.info("User is choosing to create a virtual environment")
            venv_dir = ".venv"
            try:
                logger.info("Creating virtual environment...")
                subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
                logger.info("Virtual environment created successfully")
                
                # instructions for manual activation
                if os.name == "nt": # Windows
                    print(f"Activate the virtual environment using: {venv_dir}\\Scripts\\activate")
                else: # macOS/Linux
                    print(f"Activate the virtual environment using: source {venv_dir}/bin/activate")

                print("After activation, re-run this script inside the virtual environment.")

                sys.exit(0) # user must activate manually for persistance
            except subprocess.CalledProcessError as e:
                logger.exception(f"Failed to create virtual environment", exc_info=e)
                sys.exit(-1)
    
    # get list of installed modules
    installed_packages = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
    # put those into a dict
    installed_dict = {
        pkg.split("==")[0]: pkg.split("==")[1]
        for pkg in installed_packages.stdout.splitlines()
        if "==" in pkg
    }
    logger.debug(f"Found installed packages: {installed_dict}")
    
    # get required packages (exclude any commented out)
    with open(REQUIREMENTS, 'r') as file:
        required_packages = [line.strip() for line in file if line.strip() and not line.startswith("#")]
    logger.debug(f"Found required packages: {required_packages}")
    
    missing_or_outdated = []
    for package in required_packages:
        # split each package into the name and version, ignoring delimiter ==
        pkg_name, _, required_version = package.partition("==")
        # only add modules that are not installed or a different version
        if pkg_name not in installed_dict or (required_version and installed_dict[pkg_name] != required_version):
            missing_or_outdated.append(package)
    logger.debug(f"The following packages need to be installed or updated: {missing_or_outdated}")
    
    if missing_or_outdated:
        try:
            subprocess.run(["pip3", "install"] + missing_or_outdated, check=True)
        except subprocess.CalledProcessError as e:
            logger.exception(f"Failed to install dependencies", exc_info=e)
            sys.exit(1)
    else:
        print("All dependencies are already installed and up-to-date.")
        logger.info("All dependencies are already installed and up-to-date")
    

