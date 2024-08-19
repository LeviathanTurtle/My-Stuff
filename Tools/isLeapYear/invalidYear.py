# Author: William Wadsworth
# Created: 3.25.2021
# Python-ized: 3.30.2024
# CSC 1720
# 
# About: 
#   This is the header and implementation file for the invalidYear class


# --- CLASS -------------------------------------
class invalidYear:
    #   <var> -> public
    #  _<var> -> protected
    # __<var> -> private
    #__message: str
    
    # constructor
    def __init__(self, message="invalid year"):
        self.__message = message
    
    # function huh returns the message from the invalidYear class precondition: message is declared
    # and initialized postcondition: message is returned to the main program
    def huh(self) -> str:
        return self.__message
    