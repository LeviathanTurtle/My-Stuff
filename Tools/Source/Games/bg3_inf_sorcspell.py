import keyboard
import pyautogui
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

# LAYOUT: 
# TARGET GOAL -> need:
# I: 20   -> +16 slots (32 pts)
Target_spellslots_1 = 20
Spell_1_pts = 32
# II: 20  -> +17 slots (51 pts)
unlocked_spellslots_2 = True
Target_spellslots_2 = 20
Spell_2_pts = 51
# III: 10 -> +7 slots (35 pts)
unlocked_spellslots_3 = True
Target_spellslots_3 = 10
Spell_3_pts = 35
# IV: 10  -> +7 slots (42 pts)
unlocked_spellslots_4 = True
Target_spellslots_4 = 10
Spell_4_pts = 42
# V: 5    -> +4 slots (28 pts)
unlocked_spellslots_5 = True
Target_spellslots_5 = 5
Spell_5_pts = 28

sorcery_pts = 30

#total: 188 pts needed + 30 pts for metamagic

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
    

print(LOOP_COUNTER)


"""
def open_notepad_macro():
    pyautogui.press('win')
    time.sleep(1)
    pyautogui.typewrite('notepad', interval=0.1)
    pyautogui.press('enter')
    time.sleep(1)
    pyautogui.typewrite('Hello, this is a macro!', interval=0.1)

# Bind the macro to a hotkey (e.g., CTRL+ALT+M)
keyboard.add_hotkey('ctrl+alt+m', open_notepad_macro)

# Keep the script running to listen for the hotkey
keyboard.wait('esc')  # Exit the script by pressing ESC
"""