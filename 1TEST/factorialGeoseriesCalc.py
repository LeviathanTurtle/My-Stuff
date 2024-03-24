# FACTORIAL AND GEOSERIES CALCULATOR -- V.PY
# William Wadsworth
# CSC1710
# Created: 10.15.2020
# Doctored: 10.15.2023
# Python-ized: 3.19.2024
#  
# This program performs a factorial or geoseries calculation based on a number given from input.


# --- IMPORTS ---------------------------------------------------------------------------
"""
include <iostream>
using namespace std;

long int factorial(int);
long int dfactorial(int);
double geoseries(double, int);
"""

# --- FUNCTIONS -------------------------------------------------------------------------
# --- FACTORIAL ---------------------------------
"""
long int factorial (int endPoint)
{
   long int prod = 1;

   for (int x=1; x <= endPoint; x++) {
      prod *= x;
   }

   return prod;
}
"""
# return long int
def factorial(endPoint: int) -> int:
    prod = 1
    
    for x in range(1,endPoint+1):
        prod *= x
    
    return prod

# --- DOUBLE FACTORIAL --------------------------
"""
long int dfactorial (int endPoint)
{
   int x = 1, prod = 1;

   do
   {
      prod *= x;
      x += 2;
   }
   while (x <= endPoint);

   return prod;
}
"""
# return long int
def dfactorial(endPoint: int) -> int:
    x =1
    prod = 1
    
    while x <= endPoint:
        prod *= x
        x += 2
    
    return prod

# --- GEOSERIES ---------------------------------
"""
double geoseries (double a, int t)
{
   double sum = 0;

   for (int count=0; count < t; count++) {
      sum += a;
      a *= .5;
   }

   return sum;
} 
"""
# note default of r=0.5
def geoseries(a: float, t: int, r=0.5) -> float:
    sum = 0
    
    for count in range(0,t):
        sum += a
        a *= r
        
    return sum

# --- MAIN ------------------------------------------------------------------------------
# --- INTRODUCTION ------------------------------
"""
int main () 
{
   long int endPoint;

   cout << "Factorial and Geometric series calculations" << endl << " " << endl;
"""
print("Factorial and Geometric series calculations")

# --- OPERATION SELECTION -----------------------
"""
   string inp;
   cout << "Enter 'factorial', 'dfactorial', or 'geometric': ";
   cin >> inp;
"""
inp = input("Select operation 'factorial', 'dfactorial', or 'geometric': ")
# input validation
while(inp != "factorial" and inp != "dfactorial" and inp != "geometric"):
    inp = input("error: invalid operation: ")

# --- FACTORIAL ---------------------------------
"""
   if (inp == "factorial") {
      cout << "Integer must be between -1,000 and 1,000" << endl << " " << endl;
      cout << "Enter an integer to the nearest whole for factorial calculation: ";
      cin >> endPoint;

      while (endPoint >= 1000 || endPoint <= -1000) {
         cout << "Not valid, integer must be between -1,000 and 1,000: ";
         cin >> endPoint;
      }

      cout << endPoint << "! = " << factorial(endPoint) << endl;
   }
"""
if(inp == "factorial"):
    # parameters for input, store in variable
    print("\nInteger must be between -1,000 and 1,000\n")
    
    endPoint = int(input("Enter an integer to the nearest whole for factorial calculation: "))
    # input validation
    while(endPoint < -1000 or endPoint > 1000):
        endPoint = int(input("Not valid, integer must be between -1,000 and 1,000: "))
    
    # ouput calculation
    print(f"{endPoint}! = {factorial(endPoint)}")

# --- DOUBLE FACTORIAL --------------------------
    """
    else if (inp == "dfactorial") {
        cout << "Integer must be between -1000 and 1000" << endl << " " << endl;
        cout << "Enter an integer to the nearest whole for double factorial calculation: ";
        cin >> endPoint;

        while (endPoint % 2 == 0 || endPoint >= 1000 || endPoint <= -1000) {
            cout << "Not valid, integer must be odd and between -1,000 and 1,000: ";
            cin >> endPoint;
        }

        cout << endPoint << "!! = " << dfactorial(endPoint) << endl;
    }
    """
elif(inp == "dfactorial"):
    # parameters for input, store in variable
    print("\nInteger must be between -1,000 and 1,000\n")
    
    endPoint = int(input("Enter an integer to the nearest whole for double factorial calculation: "))
    # input validation
    while(endPoint % 2 == 0 or endPoint < -1000 or endPoint > 1000):
        endPoint = int(input("Not valid, integer must be odd and between -1,000 and 1,000: "))
    
    # output calculation
    print(f"{endPoint}!! = {dfactorial(endPoint)}")

# --- GEOSERIES ---------------------------------
    """
    else {
        double a, r = 0.5, t, sum = 0;
        int count;

        cout << "Sum of Geometric series" << endl << " " << endl;
        cout << "What is your first term: ";
        cin >> a;

        cout << "How many terms would you like to take the sum of: ";
        cin >> t;

        cout << "Sum of " << t << " terms, a = " << a << ", r = 0.5, is " << geoseries(a, t) << endl;
    }

   return 0;
}
    """
# assume geoseries
else:
    print("Sum of Geometric series")
    
    a = float(input("What is your first term: "))    
    t = float(input("How many terms would you like to take the sum of: "))
    r = float(input("Enter your common ratio of choice (leave blank for 0.5): "))
    
    print(f"Sum of {t} terms, a = {a}, r = 0.5, is {geoseries(a,t,r)}")

