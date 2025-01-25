
# 
# William Wadsworth
# 1.23.2025
# 
# 
# [notes]
# 

import keyboard
from pydirectinput import moveTo, mouseDown, mouseUp
from time import sleep, time

#MOUSE_SPEED = 1 
# ^ pyautogui uses this, but not pydirectinput

# the length of time to sleep between inputs (in seconds)
SLEEP_DURATION = .05

# -------------------------------------------------------------------------------------------------

def move_and_click(coord_x: int, coord_y: int):
    """Moves the mouse to an onscreen coordinate and clicks."""
    
    moveTo(coord_x, coord_y)
    sleep(SLEEP_DURATION/2) # small buffer
    
    mouseDown()
    sleep(0.1) # hold for 100ms
    mouseUp()
    
    #sleep(SLEEP_DURATION)

def select_metamagic(type: str):
    """Select the metamagic icons in the UI."""
    
    if type == "SORCPTS": # 1745 1215
        move_and_click(1745,1215)
    elif type == "SPELLSLOTS": # 1745 1275
        move_and_click(1745,1275)
    else:
        pass # error

def create_sorcery_pts(spellslot_level: int):
    """function def."""
    
    #print(f"Creating {spellslot_level} sorcery pt(s)...")
    
    spellslot_level_y = 1250
    match (spellslot_level):
        case 1:
            # 1190 1250
            spellslot_level_x = 1190
        case 2:
            # 1250 1250
            spellslot_level_x = 1250

    select_metamagic("SORCPTS")

    # select spell slot level
    move_and_click(spellslot_level_x,spellslot_level_y)
    
    # cast (mouse pos 1200 1050)
    move_and_click(1200,1050)
    
    #print(f"Created {spellslot_level} sorcery pt(s)")
    sleep(2)

def create_spellslot(spellslot_level: int):
    """function def."""
    
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
    
    # goto spellslot icon (mouse pos 1745 1275)
    select_metamagic("SPELLSLOTS")
    
    # goto spellslot level (dynamic mouse pos)
    move_and_click(spellslot_level_x, spellslot_level_y)
    
    # cast (mouse pos 1200 1050 minimum)
    #pydirectinput.moveTo(1200, 950)
    #time.sleep(SLEEP_DURATION/2)
    move_and_click(1200,950)
    #time.sleep(SLEEP_DURATION)
    
    sleep(2)

def activate_equipment(using_shield: bool):
    """function def."""
    
    # EVENT LOOP:
    # 
    # move mouse (1285 1330)
    # click (equip item)
    # create_sorc_pts
    # move mouse
    # click (unequip item)
    
    #print("Equipping item")
    move_and_click(1285,1330)
    
    # sorc pts
    if using_shield:
        create_sorcery_pts(1)
    else:
        create_sorcery_pts(2) # todo: amulet pts count as 2
    
    #print("Unequipping item")
    move_and_click(1285,1330)
    
    sleep(.025)

def macro():
    """function def."""
    
    #############################################
    # CHANGE ALL CURRENT VALUES HERE
    
    using_shield = True           # set to false if using the amulet
    unlocked_spellslots_3 = True # set any of these to true if they are unlocked and you want 
    unlocked_spellslots_4 = True # them expanded
    unlocked_spellslots_5 = True
    
    # UPDATE THESE VALUES TO REFLECT YOUR CURRENT SPELL SLOTS
    # Note that this number should be considered after all sorc pts are gathered (since we are 
    # using first and second level spell slots) 
    
    if using_shield: # if we use the shield, we will use all first-level spell slots
        Current_spellslots_1 = 0 # leave this 0
        Current_spellslots_2 = 20
    else: # likewise for the amulet, we will use all second-level spell slots
        Current_spellslots_1 = 4
        Current_spellslots_2 = 0 # leave this 0
    Current_spellslots_3 = 3
    Current_spellslots_4 = 3
    Current_spellslots_5 = 1
    Current_sorc_pts = 51
    
    # THESE ARE THE TARGET VALUES YOU WANT
    Target_spellslots_1 = 20 # e.g. I want 20 first-level spell slots
    Target_spellslots_2 = 20
    Target_spellslots_3 = 10
    Target_spellslots_4 = 10
    Target_spellslots_5 = 5
    Target_sorc_pts = 30
    #############################################

    # SETUP: 
    # TARGET GOAL -> need:
    # I: 20   -> +16 slots (32 pts) [x2]
    Target_spellslots_1 -= Current_spellslots_1
    Spell_1_pts = Target_spellslots_1*2
    # Note: if we're using the shield, all of these will be depleted so we need to create all of them

    # II: 20  -> +17 slots (51 pts) [x3]
    Target_spellslots_2 -= Current_spellslots_2
    Spell_2_pts = Target_spellslots_2*3
    # Note: if we're using the amulet, all of these will be depleted so we need to create all of them
    Target_sorc_pts -= Current_sorc_pts 

    # the loop counter needs to be the total amount of sorcery points needed
    LOOP_COUNTER: int = Spell_1_pts+Spell_2_pts + Target_sorc_pts
    # III: 10 -> +7 slots (35 pts) [x5]
    if unlocked_spellslots_3:
        Target_spellslots_3 -= Current_spellslots_3
        Spell_3_pts = Target_spellslots_3*5
        LOOP_COUNTER += Spell_3_pts
        
        # IV: 10  -> +7 slots (42 pts) [x6]
        if unlocked_spellslots_4:
            Target_spellslots_4 -= Current_spellslots_4
            Spell_4_pts = Target_spellslots_4*6
            LOOP_COUNTER += Spell_4_pts
            
            # V: 5    -> +4 slots (28 pts) [x7]
            if unlocked_spellslots_5:
                Target_spellslots_5 -= Current_spellslots_5
                Spell_5_pts = Target_spellslots_5*7
                LOOP_COUNTER += Spell_5_pts

    print(f"Need {LOOP_COUNTER} more sorc pts ({LOOP_COUNTER+Current_sorc_pts} total)")
    
    sleep(5)
    start_time = time()
    
    # -----------------
    
    # get all necessary sorc pts
    msg = f"Getting {LOOP_COUNTER} sorc pts ({Target_sorc_pts} pt targ + {Spell_1_pts} pts for lvl 1 spellslots + {Spell_2_pts} pts for lvl 2 spellslots"
    if unlocked_spellslots_3:
        msg += f" + {Spell_3_pts} pts for lvl 3 spellslots"
        if unlocked_spellslots_4:
            msg += f"  {Spell_4_pts} pts for lvl 4 spellslots"
            if unlocked_spellslots_5:
                msg += f" + {Spell_5_pts} pts for lvl 5 spellslots"
    print(msg+")...")
    
    for _ in range(LOOP_COUNTER):
        activate_equipment(using_shield)
    
    # expand spell slots
    spellslot_data = {
        1: {"target": Target_spellslots_1, "unlocked": True},
        2: {"target": Target_spellslots_2, "unlocked": True},
        3: {"target": Target_spellslots_3, "unlocked": unlocked_spellslots_3},
        4: {"target": Target_spellslots_4, "unlocked": unlocked_spellslots_4},
        5: {"target": Target_spellslots_5, "unlocked": unlocked_spellslots_5},
    }

    # loop through levels and create spellslots if unlocked
    for level, data in spellslot_data.items():
        if data["unlocked"]:  # check if the level is unlocked
            print(f"Creating {data['target']} lvl {level} spellslots...")
            for _ in range(data["target"]):
                create_spellslot(level)
    
    stop_time = time()
    print(f"Macro completed in {stop_time-start_time:.2f}s")


def main():
    #time.sleep(5)
    # Bind the macro to a hotkey (e.g., CTRL+ALT+M)
    keyboard.add_hotkey('shift+r', macro)
    
    # Keep the script running to listen for the hotkey
    keyboard.wait('esc')  # Exit the script by pressing ESC


if __name__ == "__main__":
    main()

