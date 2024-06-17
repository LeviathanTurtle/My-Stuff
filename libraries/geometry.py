# Author: William Wadsworth
# Date: 6.16.2024
# 
# About:
#     This is the implementation file for the geometry class

from math import sqrt, pi

class geometry:
    """idk"""
    
    a: float
    b: float
    c: float
    angle_a: float
    angle_b: float
    angle_c: float
    point_x: float
    point_y: float
    point_z: float
    cir_radius: float
    cir_diameter: float = 2*cir_radius
    cir_circumference: float
    
    # function to calculate the distance between two cartesian coordinates (x1,y1,x2,y2)
    # pre-condition: the coordinates (passed as (x1,y1,x2,y2)) should be initialized as valid
    #                cartesian coordinate points
    # 
    # post-condition: the distance between the two points is returned
    def distance(q: float, w: float, e: float, r: float) -> float:
        return sqrt((e-q)**2 + (r-w)**2)
    
    # function to calculate the radius between two cartesian coordinates (x1,y1,x2,y2)
    # pre-condition: the coordinates (passed as (x1,y1,x2,y2)) should be initialized as valid
    #                cartesian coordinate points 
    # 
    # post-condition: the radius between the two points is returned
    def radius(self, a: float, s: float, d: float, f: float) -> float:
        return self.distance(a,s,d,f)
    
    # function to determine the circumference of a circle
    # pre-condition: the radius must be initialized to a positive non-zero float
    # 
    # post-condition: the calculated circumference is returned
    def circumference(radius: float) -> float:
        return 2*pi*(radius**2)
    
    # function to calculate the area of a circle
    # pre-condition: the radius must be initialized to a positive non-zero float
    # 
    # post-condition: the calculated area is returned
    def area_circle(radius: float) -> float:
        return pi*(radius**2)
    
    # function to determine if the sides of a triangle make up a right triangle
    # pre-condition: sides parameters a, b, and c must be initialized to positive non-zero floats.
    #                The hypotenuse should be side c
    # 
    # post-condition: if the triangle is a right triangle, true is returned, otherwise false
    def isRightTriangle(self) -> bool:
        if (self.a*self.a) + (self.b*self.b) == (self.c*self.c):
            return True
        else:
            return False

