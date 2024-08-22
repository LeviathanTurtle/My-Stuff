# GRIDPOINT OPERATIONS -- V.PY
# William Wadsworth
# CSC1710
# Created: 10.22.2020
# Dcotored: 11.2.2023
# 
# Python-ized: 3.17.2024
# Updated 8.18.2024: PEP 8 Compliance
# 
# [DESCRIPTION]:
# This program prompts the user for two coordinate points, and the operation the user would like to
# perform. In the context of ellipsoids, this program assumes you are working with a circle.
# 
# [USAGE]:
# To run: python3 gridpointOperations.py


from sys import exit
from math import sqrt

PI: float = 3.141592


# --- FUNCTIONS -------------------------------------------------------------------------
# --- DISTANCE ----------------------------------
# pre-condition: q, w, e, and r must be real numbers. If e and r are None, the function calculates
#                the distance from the origin (0,0) to (q, w). If e and r are provided, the
#                function calculates the distance between (q, w) and (e, r)
# post-condition: returns the Euclidean distance between the specified points
def distance(q: float, w: float, e: float = None, r: float = None) -> float:
    """Calculate the distance between two points."""
    
    if e is None and r is None:
        return sqrt(q**2 + w**2)
    else:
        return sqrt((e-q)**2 + (r-w)**2)


# --- RADIUS ------------------------------------
# pre-condition: a, s, d, and f must be real numbers
# post-condition: returns the radius of the circle
def radius(a: float, s: float, d: float, f: float) -> float:
    """Calculate the radius of a circle using two points."""
    
    inp = input("Is the line between these two points a diameter? [Y/n]: ")
    # input validation
    while inp not in ("Y", "n"):
        inp = input("error: not a valid response [Y/n]: ")
        
    if inp == "Y":
        return distance(a,s,d,f) / 2
    else:
        return distance(a,s,d,f)


# --- CIRCUMFERENCE -----------------------------
# pre-condition: r must be a non-negative real number
# post-condition: returns the circumference of the circle
def circumference(r: float) -> float:
    """Calculate the circumference of a circle given its radius."""
    
    return 2 * PI * r


# --- AREA --------------------------------------
# pre-condition: r must be a non-negative real number
# post-condition: returns the area of the circle
def area(r: float) -> float:
    """Calculate the area of a circle given its radius."""
    
    return PI * (r**2)


def main():
    # --- INTRODUCTION --------------------------
    # input requirements
    print("Points must be entered as two integers, such as: x y")

    # --- FIRST POINT ---------------------------
    x1, y1 = map(float, input("Input your first point: ").split())

    # --- SECOND POINT ---------------------------
    x2, y2 = map(float, input("Input your second point: ").split())

    # --- OPERATION SELECTION -------------------
    # output decimals, ask for intended operation
    print("Possible operations: distance, radius, circumference, area")
    inp = input("What operation would you like to do: ")
    # input validation
    while inp not in ("distance", "radius", "circumference", "area"):
        inp = input("error: invalid operation: ")

    # --- OPERATION CALLS -----------------------
    # call functions based on input
    if inp == "distance":
        print(f"The distance between the two points is {distance(x1,y1,x2,y2):.3f} units.")
    elif inp == "radius":
        print(f"The radius is {radius(x1,y1,x2,y2):.3f} units.")
    elif inp == "circumference":
        print(f"The circumference is {circumference(radius(x1,y1,x2,y2)):.3f} units.")
    elif inp == "area":
        print(f"The area is {area(radius(x1,y1,x2,y2)):.3f} units squared.")
    # Note: due to input validation in previous section (OPERATION SELECTION), this else clause
    # should not be hit.
    else:
        print("whoop de do, I have no clue")
        exit(1)


if __name__ == "__main__":
    main()