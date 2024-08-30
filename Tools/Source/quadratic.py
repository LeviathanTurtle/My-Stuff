# solves the quadratic equation

from math import sqrt

# input coefficients
a = float(input("Enter coefficient a: "))
b = float(input("Enter coefficient b: "))
c = float(input("Enter coefficient c: "))

# calculate discriminant
discriminant = b**2 - 4*a*c

# 
if discriminant > 0:
    # two real and distinct roots
    root1 = (-b + sqrt(discriminant)) / (2*a)
    root2 = (-b - sqrt(discriminant)) / (2*a)
    print(f"Root 1: {root1}\nRoot 2: {root2}")
elif discriminant == 0:
    # one real root
    root = -b / (2*a)
    print(f"Root: {root}")
else:
    # complex roots
    real_part = -b / (2*a)
    imaginary_part = sqrt(abs(discriminant)) / (2*a)
    print(f"Root 1: {real_part} + {imaginary_part}i\nRoot 2: {real_part} - {imaginary_part}i")