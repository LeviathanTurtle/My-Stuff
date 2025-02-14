# Author: William Wadsworth
# Date: 6.16.2024
# 
# About:
#     This is the implementation file for the Python geometry class

from math import sqrt, pi

class Geometry:
    """idk"""
    
    # pre-condition: the coordinates (passed as (x1,y1,x2,y2)) should be initialized as valid
    #                cartesian coordinate points
    # 
    # post-condition: the distance between the two points is returned
    @staticmethod
    def distance(q: float, w: float, e: float, r: float) -> float:
        """Calculates the distance between two cartesian coordinates (x1,y1,x2,y2)."""
        
        return sqrt((e-q)**2 + (r-w)**2)
    
    # pre-condition: the coordinates (passed as (x1,y1,x2,y2)) should be initialized as valid
    #                cartesian coordinate points 
    # 
    # post-condition: the radius between the two points is returned
    @staticmethod
    def radius(a: float, s: float, d: float, f: float) -> float:
        """Calculates the radius between two cartesian coordinates (x1,y1,x2,y2)."""
        
        return Geometry.distance(a,s,d,f)
    
    # pre-condition: the radius must be initialized to a positive non-zero float
    # 
    # post-condition: the calculated circumference is returned
    @staticmethod
    def circumference(radius: float) -> float:
        """Calculates the circumference of a circle."""
        
        return 2*pi*(radius**2)
    
    # pre-condition: the radius must be initialized to a positive non-zero float
    # 
    # post-condition: the calculated area is returned
    @staticmethod
    def area_circle(radius: float) -> float:
        """Calculates the area of a circle."""
        
        return pi*(radius**2)
    
    # pre-condition: a, b, and c are real numbers (floats) greater than zero. The hypotenuse should
    #                be side c
    # post-condition: if the squares of the two sides equal the square of the hypotenuse, True is
    #                 returned, otherwise False
    @staticmethod
    def isRightTriangle(
        a: float,
        b: float,
        c: float
    ) -> bool:
        """Determines if three sides of a triangle make a right triangle."""

        return True if (a*a)+(b*b)==(c*c) else False

