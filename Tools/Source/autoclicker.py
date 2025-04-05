# 
# William Wadsworth
# 

from argparse import ArgumentParser
from pydirectinput import mouseDown, mouseUp
from keyboard import add_hotkey, wait
from time import sleep
import sys

EXIT_HOTKEY = 'esc'

def autoclicker(delay: int):
    while True:
        mouseDown()
        sleep(delay) # hold for 10ms
        mouseUp()

def main():
    parser = ArgumentParser(description="Autoclicker :).")
    parser.add_argument("-delay", help="Delay (in ms).", type=float)
    
    args = parser.parse_args()
    DELAY = args.delay
    
    # small delay before beginning
    sleep(5)
    
    try:
        # add hotkey
        add_hotkey(EXIT_HOTKEY, sys.exit)
        autoclicker(DELAY)
        
        # keep the script running to listen for the hotkey
        wait()
    except KeyboardInterrupt:
        print("Autoclicker stopped.")

if __name__ == "__main__":
    main()

