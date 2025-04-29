
# 
# William Wadsworth
# 2.9.2025
# 
# Use this to test the mouse coordinates for the U.I.
# 

from pydirectinput import moveTo
from dotenv import load_dotenv, set_key
from os import path
from config import *

WORKSPACE_ROOT = "../"
ENV_FILE = path.join(WORKSPACE_ROOT, ".env")
set_key(ENV_FILE, "CHECKED_MOUSE_COORDS", "False")

def main() -> None:
    load_dotenv()
    print() # for extra space
    
    for name, spot in COORDINATE_MAP.items():
        #x, y, *others = spot # unpack first two values as x, y, remaining in others
        print(f"\n{name.value}: {spot.x}, {spot.y}")
        moveTo(spot.x,spot.y)
        
        #if len(others) > 0:
        #    print(f"On line(s): {others}")
        
        input("Press Enter to continue...")
    
    set_key(ENV_FILE, "CHECKED_MOUSE_COORDS", "True")
    print("Done")


if __name__ == "__main__": 
    main()