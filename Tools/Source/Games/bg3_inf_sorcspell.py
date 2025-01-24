
# 
# William Wadsworth
# 1.23.2025
# 

import keyboard
#import pyautogui
import pydirectinput
import time

# INFORMATION:
"""
CONVERSIONS:
SLOTS:
I: 2 pts
II: 3 pts
III: 5 pts
IV: 6 pts
V: 7 pts

SORCERY PTS based on spell level (e.g. 3 = 3 pts)
"""

Base_spellslots_1 = 4
Base_spellslots_2 = 3
Base_spellslots_3 = 3
Base_spellslots_4 = 3
Base_spellslots_5 = 1
using_shield_of_devotion = True
using_spell_savant_amulet = False
#MOUSE_SPEED = 1
SLEEP_DURATION = .4

# LAYOUT: 
# TARGET GOAL -> need:
# I: 20   -> +16 slots (32 pts)
Target_spellslots_1 = 20
Spell_1_pts = 32

# II: 20  -> +17 slots (51 pts)
unlocked_spellslots_2 = True
Target_spellslots_2 = 20
Spell_2_pts = 51
sorcery_pts = 30

# III: 10 -> +7 slots (35 pts)
unlocked_spellslots_3 = False
Target_spellslots_3 = 10
Spell_3_pts = 35

# IV: 10  -> +7 slots (42 pts)
unlocked_spellslots_4 = False
Target_spellslots_4 = 10
Spell_4_pts = 42

# V: 5    -> +4 slots (28 pts)
unlocked_spellslots_5 = False
Target_spellslots_5 = 5
Spell_5_pts = 28

# the loop counter needs to be the total amount of sorcery points needed
LOOP_COUNTER: int = Spell_1_pts
if unlocked_spellslots_2:
    LOOP_COUNTER += Spell_2_pts + sorcery_pts
    
    if unlocked_spellslots_3:
        LOOP_COUNTER += Spell_3_pts
        if unlocked_spellslots_4:
            LOOP_COUNTER += Spell_4_pts
            if unlocked_spellslots_5:
                LOOP_COUNTER += Spell_5_pts
    

#print(LOOP_COUNTER)


"""
def open_notepad_macro():
    pydirectinput.press('win')
    time.sleep(1)
    pydirectinput.typewrite('notepad', interval=0.1)
    pydirectinput.press('enter')
    time.sleep(1)
    pydirectinput.typewrite('Hello, this is a macro!', interval=0.1)

# Bind the macro to a hotkey (e.g., CTRL+ALT+M)
keyboard.add_hotkey('ctrl+alt+m', open_notepad_macro)

# Keep the script running to listen for the hotkey
keyboard.wait('esc')  # Exit the script by pressing ESC
"""


# -------------------------------------------------------------------------------------------------

def mouse_click():
    pydirectinput.mouseDown()
    time.sleep(0.1)  # Hold for 100ms
    pydirectinput.mouseUp()

def select_metamagic(type: str):
    """function def."""
    
    if type == "SORCERY PTS": # 1745 1215
        pydirectinput.moveTo(1745, 1215)
        time.sleep(SLEEP_DURATION/2)
        mouse_click()
        time.sleep(SLEEP_DURATION)
    elif type == "SPELL SLOTS": # 1745 1275
        pydirectinput.moveTo(1745, 1275)
        time.sleep(SLEEP_DURATION/2)
        mouse_click()
        time.sleep(SLEEP_DURATION)
    else:
        pass # error

def create_sorcery_pts(spellslot_level: int):
    """function def."""
    
    print(f"Creating {spellslot_level} sorcery pt(s)...")
    
    spellslot_level_y = 1250
    match (spellslot_level):
        case 1:
            # 1190 1250
            spellslot_level_x = 1190
        case 2:
            # 1250 1250
            spellslot_level_x = 1250
        case 3:
            # 1300 1250 
            spellslot_level_x = 1300
        case 4:
            # 1365 1250
            spellslot_level_x = 1365
        case 5:
            # 1425 1250
            spellslot_level_x = 1425

    select_metamagic("SORCERY PTS")

    # select spell slot level
    pydirectinput.moveTo(spellslot_level_x, spellslot_level_y)
    time.sleep(SLEEP_DURATION/2)
    mouse_click()
    time.sleep(SLEEP_DURATION)
    
    # cast mouse pos: 1200 1050
    pydirectinput.moveTo(1200, 950)
    time.sleep(SLEEP_DURATION/2)
    mouse_click()
    #time.sleep(SLEEP_DURATION)
    
    print(f"Created {spellslot_level} sorcery pts")

def create_spellslot(spellslot_level: int):
    pass

def activate_shield():
    """function def."""
    
    # EVENT LOOP (shield):
    # 
    # move mouse (1285 1330)
    # click (equip shield)
    # create_sorc_pts
    # move mouse
    # click (unequip shield)
    
    print("Equipping shield")
    pydirectinput.moveTo(1285, 1330)
    time.sleep(SLEEP_DURATION/2)
    mouse_click()
    time.sleep(SLEEP_DURATION)
    
    # sorc pts
    create_sorcery_pts(1)
    time.sleep(2.4)
    
    print("Unequipping shield")
    pydirectinput.moveTo(1285, 1330)
    time.sleep(SLEEP_DURATION/2)
    mouse_click()
    time.sleep(SLEEP_DURATION)

def macro():
    """function def."""
    
    time.sleep(5)
    #create_sorcery_pts(1)
    
    # -----------------
    for _ in range(Base_spellslots_1):
        create_sorcery_pts(1)
        time.sleep(2.4)
    
    # get all necessary sorcery pts
    for _ in range(LOOP_COUNTER):
        activate_shield()
        time.sleep(.75)
    
    # expand spell slots


def main():
    #time.sleep(5)
    # Bind the macro to a hotkey (e.g., CTRL+ALT+M)
    keyboard.add_hotkey('shift+r', macro)
    
    # Keep the script running to listen for the hotkey
    keyboard.wait('esc')  # Exit the script by pressing ESC
    keyboard.add_hotkey('esc', keyboard.wait)


if __name__ == "__main__":
    main()


# EVENT LOOP (amulet):
# 
# cast all lvl 2
# loop:
#   move mouse
#   click (equip amulet)
#   move mouse
#   click (activate metamagic)
#   move mouse
#   click (select level)
#   move mouse
#   click (cast)

