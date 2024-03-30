# Author: William Wadsworth
# Created: 3.25.2021
# Python-ized: 3.30.2024
# CSC 1720
# 
# About: 
#   This is the header and implementation file for the invalidMonth class

# --- IMPORTS -----------------------------------
"""
#include <iostream>
using namespace std;
"""

# --- CLASS -------------------------------------
"""
class invalidMonth
{
   private:
      string message;
   public:
      invalidMonth()
      {
         message = "invalid month";
      }

      invalidMonth(string str)
      {
         message = str;
      }
      
      string huh()
      {
         return message;
      }
};
"""
class invalidMonth:
    #   <var> -> public
    #  _<var> -> protected
    # __<var> -> private
    #__message: str
    
    # constructor
    def __init__(self, message="invalid month"):
        self.__message = message
    
    # function huh returns the message from the invalidMonth class precondition: message is
    # declared and initialized postcondition: message is returned to the main program
    def huh(self) -> str:
        return self.__message
    