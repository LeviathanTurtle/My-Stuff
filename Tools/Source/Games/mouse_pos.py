import pyautogui
import time

print("Move your mouse to a desired position. Press Ctrl+C to stop.")

try:
    while True:
        # Get the current mouse position
        x, y = pyautogui.position()
        print(f"Mouse position: ({x}, {y})", end="\r")  # Print on the same line
        time.sleep(0.1)  # Slight delay to make it readable
except KeyboardInterrupt:
    print("\nMouse position capturing stopped.")
