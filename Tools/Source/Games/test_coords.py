
# 
# William Wadsworth
# 2.9.2025
# 
# Use this to test the mouse coordinates for the U.I.
# 

from pydirectinput import moveTo


def main() -> None:
    
    ui_coords = {
        "Equipment": (1285,1330),
        "Freecast": (1660,1160),
        "Sorcery points": (1745,1215),
        "Spell slots": (1745,1275),
    }
    
    for name, spot in ui_coords.items():
        print(f"\n{name}: {spot}")
        moveTo(spot[0], spot[1])
        input("Press Enter to continue...")
    
    print("Done")


if __name__ == "__main__":
    main()