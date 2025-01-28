
# 
# William Wadsworth
# 1..2025
# 

from pyautogui import position
from time import sleep

print("Move your mouse to a desired position. Press Ctrl+C to stop.")

try:
    while True:
        x, y = position() # get the current mouse position
        print(f"Mouse position: ({x}, {y})", end="\r") # print on the same line
        sleep(0.1) # slight delay to make it readable
except KeyboardInterrupt:
    print("\nMouse position capturing stopped.")
