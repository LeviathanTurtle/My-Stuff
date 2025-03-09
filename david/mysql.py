
# 
# William Wadsworth
# 03.08.2025
# 

import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

from password import PasswordManager
#import pandas as pd

# pre-condition: 
# post-condition: 
def create_db_connection(
    host_name,
    user_name,
    user_password,
    db_name
):
    """function def."""
    
    connection = None
    try:
        connection = mysql.connector.connect(
            host=host_name,
            user=user_name,
            passwd=user_password,
            database=db_name
        )
        print("MySQL database connection successful")
    except Error as err:
        print(f"Error: {err}")
    
    return connection

# pre-condition: 
# post-condition: 
def create_query(
    connection,
    query: str
) -> None:
    """function def."""
    
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        print("Database created successfully")
    except Error as err:
        print(f"Error: {err}")

# pre-condition: 
# post-condition: 
def execute_query(
    connection,
    query
):
    """function def."""
    
    cursor = connection.cursor()
    try:
        cursor.execute(query)
        connection.commit()
        print("Query successful")
    except Error as err:
        print(f"Error: {err}")



def main():
    
    if load_dotenv():
        password_manager = PasswordManager()
        print()

        connection = create_db_connection("localhost","root",password_manager.get_password())
    else:
        print("load_dotenv error")

if __name__ == '__main__':
    main()