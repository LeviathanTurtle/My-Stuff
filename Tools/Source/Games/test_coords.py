
# 
# William Wadsworth
# 2.9.2025
# 
# Use this to test the mouse coordinates for the U.I.
# 

from pydirectinput import moveTo


def main() -> None:
    
    ui_coords = {
        "Equipment": (1285,1330, 302,311),
        "Freecast": (1660,1160, 204),
        "Clothing": (1660,1220, 201),
        "Sorcery points": (1745,1215, 254),
        "Spell slots": (1745,1275, 256),
    }
    
    print()
    for name, spot in ui_coords.items():
        x, y, *others = spot # unpack first two values as x, y, remaining in others
        print(f"\n{name}: {x}, {y}")
        moveTo(x,y)
        
        if len(others) > 0:
            print(f"On line(s): {others}")
        
        input("Press Enter to continue...")
    
    print("Done")


if __name__ == "__main__":
    main()