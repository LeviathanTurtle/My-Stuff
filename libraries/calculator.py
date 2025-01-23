# Author: William Wadsworth
# Date: 6.16.2024
# Updated 8.30.2024: added quadratic function
# 
# About:
#     This is the implementation file for the calculator class

from math import sqrt
from typing import List, Tuple, Union

class Calculator:
    """A simple calculator class with various mathematical methods."""
    
    # pre-condition: endpoint should be initialized to an integer between 0 and 1,000 and
    #                double_factorial must be initialized to true of false
    # post-condition: if the user's endpoint is valid, its calculated result is returned, otherwise
    #                 -1 is returned
    @staticmethod
    def factorial(
        endpoint: int,
        double_factorial: bool
    ) -> int:
        """Returns the factorial or double factorial of a number."""
        
        # check that the user's endpoint is valid. Will probably update this later to be more
        # dynamic
        if endpoint < 0 or endpoint > 1000:
            raise ValueError("Invalid endpoint, integer must be between 0 and 1,000")

        prod: int = 1
        
        # user wants a double factorial
        if double_factorial:
            # additional check unique to double factorials
            if endpoint%2 == 0:
                raise ValueError("Invalid endpoint, integer must be between 0 and 1,000")
                #return -1
            
            for i in range(1,endpoint+1,2):
                prod *= i
        # normal factorial
        else:
            for i in range(1,endpoint+1):
                prod *= i
        
        return prod
    
    # pre-condition: starting value a must be initialized to a non-zero float, number of terms must
    #                be initialized to a non-zero integer, r must be initialized to a positive 
    #                non-zero float, but has a default of 0.5 if it is not provided or invalid
    # post-condition: the series sum is returned
    @staticmethod
    def geoseries(
        initial_term: float, 
        num_terms: int, 
        r: float = 0.5
    ) -> float:
        """Returns the geometric series of a number."""
        
        # check common ratio of choice
        if r <= 0:
            raise ValueError("Common ratio must be positive.")
            
        series_sum: float = 0.0
        term = initial_term
        
        for _ in range(num_terms):
            series_sum += term
            term *= r
        
        return series_sum
    
    # pre-condition: operand_1 and operand_2 parameters must be initialized with values. If 
    #                dividing, operand_2 cannot be 0. operation parameter must be initialized to a
    #                non-empty string
    # post-condition: depending on the operation specified (assuming the operation is valid), the
    #                 sum, difference, product, or quotient is returned, otherwise an error is
    #                 output and a relevant exception is thrown
    @staticmethod
    def fourFunction(
        operand_1: Union[int, float], 
        operand_2: Union[int, float], 
        operation: str
    ) -> Union[int, float]:
        """Returns the sum, difference, product, or quotient of two numbers (supports only
        numerical types)."""
        
        # check that both operands are integers or floats
        if not (isinstance(operand_1, (int, float)) and isinstance(operand_2, (int, float))):
            raise TypeError("Operands must be an arithmetic type")

        operation = operation.lower()

        if operation == "add":
            return operand_1 + operand_2
        elif operation == "subtract":
            return operand_1 - operand_2
        elif operation == "multiply":
            return operand_1 * operand_2
        elif operation == "divide":
            if operand_2 == 0:
                raise ValueError("Error: cannot divide by 0")
            return operand_1 / operand_2
        else:
            raise ValueError("Error: invalid operation")
        
    # pre-condition: max and increment must be initialized to positive non-zero numerical values,
    #                count must be initialized (will be reset to 0 in case of invalid value),
    #                multiples must be intialized to an empty vector
    # post-condition: if there are no multiples, false is returned, otherwise a tuple is returned
    #                 containing True, the amount of multiples, and a list of each multiple
    @staticmethod
    def isMultiple(
        max_val: Union[int, float], 
        increment: Union[int, float]
    ) -> Tuple[bool, int, List[Union[int, float]]]:
        """Determines if one number is a multiple of another (and its multiples)."""
        
        # check that max and increment are integers or floats
        if not (isinstance(max, (int, float)) and isinstance(increment, (int, float))):
            raise TypeError("max and increment must be numeric types")
        
        # ensure increment is not zero to avoid infinite loop
        if increment == 0:
            raise ValueError("increment must not be zero")
    
        # sum var for calculation
        sum = 0
        # ensure count starts at 0
        count = 0
        # define empty list
        multiples = []
        current_val = increment
        
        # calculation of multiples
        while current_val < max_val:
            multiples.append(current_val)
            current_val += increment

        return (len(multiples) > 0, len(multiples), multiples)

    # pre-condition: a, b, and c are real numbers (floats), a must not be zero
    # post-condition: if the discriminant is positive, returns two distinct real roots as a tuple
    #                 of floats. If the discriminant is zero, returns one real root as a float. If
    #                 the discriminant is negative, returns two complex roots as a tuple of complex
    #                 numbers.
    @staticmethod
    def quadratic(
        a: float, 
        b: float, 
        c: float
    ) -> Union[Tuple[float, float], float, Tuple[complex, complex]]:
        """Solves the quadratic equation."""
        
        if a == 0:
            raise ValueError("The value of 'a' cannot be zero in a quadratic equation.")
        
        # calculate discriminant
        discriminant = b**2 - 4*a*c
        
        # this determines what type of root(s) we have
        if discriminant > 0:
            # two roots
            root1 = (-b + sqrt(discriminant)) / (2*a)
            root2 = (-b - sqrt(discriminant)) / (2*a)
            return root1, root2
            #return (-b + sqrt(discriminant)) / (2*a), (-b - sqrt(discriminant)) / (2*a)
        elif discriminant == 0:
            root = -b / (2*a)
            return root
            #return -b / (2*a)
        else:
            # one part is imaginary
            real_part = -b / (2*a)
            imaginary_part = sqrt(abs(discriminant)) / (2*a)
            root1 = complex(real_part,imaginary_part)
            root2 = complex(real_part,-imaginary_part)
            return root1, root2
            #return complex(-b / (2*a),sqrt(abs(discriminant)) / (2*a)), complex(-b / (2*a),-sqrt(abs(discriminant)) / (2*a))
    
