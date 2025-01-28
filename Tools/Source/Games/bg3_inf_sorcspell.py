
# 
# William Wadsworth
# 1.23.2025
# 
# 
# [notes]
# todo PROBLEMS:
# - check conv for spellslot lvl 6
# - test with other FPS values
# - add photos in readme doc
# - graceful stoppage
# - literally test it
# 

from keyboard import add_hotkey, wait, is_pressed
from pydirectinput import moveTo, mouseDown, mouseUp
from time import sleep, time

# pyautogui uses this, but not pydirectinput
#MOUSE_SPEED = 1 
# the length of time to sleep between inputs (in seconds)
SLEEP_DURATION = .05

#################################################
# CHANGE ONLY THESE VALUES HERE

PAUSE_HOTKEY = 'p'
EXIT_HOTKEY = 'esc'

# set any of these to true if they are unlocked and you want them expanded
unlocked_spellslots_2 = True
unlocked_spellslots_3 = True
unlocked_spellslots_4 = False 
unlocked_spellslots_5 = False
unlocked_spellslots_6 = False
# set this to false if using the amulet
using_shield = True

# UPDATE THESE VALUES TO REFLECT YOUR CURRENT SPELL SLOTS
# Note that the current number of spellslots for level 1 should be 0 if using the shield, likewise
# for level 2 if using the amulet
if using_shield:
    Current_spellslots_1 = 0 # leave this 0
    Current_spellslots_2 = 3
else:
    Current_spellslots_1 = 4
    Current_spellslots_2 = 0 # leave this 0
Current_spellslots_3 = 3
Current_spellslots_4 = 1
Current_spellslots_5 = 1
Current_spellslots_6 = 1
Current_sorc_pts = 5

# THESE ARE THE TARGET VALUES YOU WANT
Target_spellslots_1 = 15 
Target_spellslots_2 = 10
Target_spellslots_3 = 10
Target_spellslots_4 = 5
Target_spellslots_5 = 5
Target_spellslots_6 = 5
Target_sorc_pts = 30

#################################################


def macro():
    """function def."""

    # SETUP:
    # LEVEL -> needed sorc pts per level:
    # I -> [x2]
    Target_spellslots_1 -= Current_spellslots_1
    Spell_1_pts = Target_spellslots_1*2

    # II -> [x3]
    Target_spellslots_2 -= Current_spellslots_2
    Spell_2_pts = Target_spellslots_2*3
    Target_sorc_pts -= Current_sorc_pts 

    # the loop counter needs to be the total amount of sorcery points needed
    LOOP_COUNTER: int = Spell_1_pts+Spell_2_pts + Target_sorc_pts
    # III -> [x5]
    if unlocked_spellslots_3:
        Target_spellslots_3 -= Current_spellslots_3
        Spell_3_pts = Target_spellslots_3*5
        LOOP_COUNTER += Spell_3_pts
        
        # IV -> [x6]
        if unlocked_spellslots_4:
            Target_spellslots_4 -= Current_spellslots_4
            Spell_4_pts = Target_spellslots_4*6
            LOOP_COUNTER += Spell_4_pts
            
            # V -> [x7]
            if unlocked_spellslots_5:
                Target_spellslots_5 -= Current_spellslots_5
                Spell_5_pts = Target_spellslots_5*7
                LOOP_COUNTER += Spell_5_pts
                
                # VI -> [x]
                #if unlocked_spellslots_6:
                #    Target_spellslots_6 -= Current_spellslots_6
                #    Spell_6_pts = Target_spellslots_6*_
                #    LOOP_COUNTER += Spell_6_pts

    # define dict to be used in main loop
    spellslot_data = {
        1: {"target": Target_spellslots_1, "unlocked": True},
        2: {"target": Target_spellslots_2, "unlocked": True},
        3: {"target": Target_spellslots_3, "unlocked": unlocked_spellslots_3},
        4: {"target": Target_spellslots_4, "unlocked": unlocked_spellslots_4},
        5: {"target": Target_spellslots_5, "unlocked": unlocked_spellslots_5},
        6: {"target": Target_spellslots_6, "unlocked": unlocked_spellslots_6},
    }

    # if we are using the amulet, update the loop counter to use half the required iterations
    if not using_shield:
        if LOOP_COUNTER % 2 != 0:
            LOOP_COUNTER //= 2 + 1 # if it is odd, offset by 1 so we are over requirement
        else: LOOP_COUNTER /= 2
        
    print(f"Need {LOOP_COUNTER} more sorc pts ({LOOP_COUNTER+Current_sorc_pts} total)")
    est_runtime_1 = ((SLEEP_DURATION/2 + .1) * 5 + 2.025) * LOOP_COUNTER
    est_runtime_2 = estimate_runtime(spellslot_data)
    print(f"Estimated runtime: {est_runtime_1+est_runtime_2:.2f}s ({(est_runtime_1+est_runtime_2)/60:.2f} min)")
    
    sleep(5)
    start_time = time()
    
    # -------------------------------------------
    
    # get all necessary sorc pts
    msg = f"Getting {LOOP_COUNTER} sorc pts ({Target_sorc_pts} pt targ + {Spell_1_pts} for lvl 1 spellslots + {Spell_2_pts} for lvl 2 spellslots"
    if unlocked_spellslots_3:
        msg += f" + {Spell_3_pts} for lvl 3 spellslots"
        if unlocked_spellslots_4:
            msg += f" + {Spell_4_pts} for lvl 4 spellslots"
            if unlocked_spellslots_5:
                msg += f" + {Spell_5_pts} for lvl 5 spellslots"
                #if unlocked_spellslots_6:
                #    msg += f" + {Spell_6_pts} for lvl 6 spellslots"
    print(msg+")...")
    
    for _ in range(LOOP_COUNTER):
        activate_equipment(using_shield)
        wait_if_paused() # pause check

    # loop through levels and create spellslots if unlocked
    for level, data in spellslot_data.items():
        if data["unlocked"]:  # check if the level is unlocked
            wait_if_paused() # pause check
            print(f"Creating {data['target']} lvl {level} spellslots...")
            for _ in range(data["target"]):
                create_spellslot(level)
                wait_if_paused() # pause check
    
    stop_time = time()
    print(f"Macro completed in {stop_time-start_time:.2f}s")

# -------------------------------------------------------------------------------------------------

def wait_if_paused():
    """Pauses the script if the pause hotkey is pressed."""
    
    while is_pressed(PAUSE_HOTKEY):
        print("Macro paused. Press 'p' again to resume.")
        sleep(1)

def estimate_runtime(spellslot_data) -> float:
    total_runtime = 0

    for level, data in spellslot_data.items():
        if data["unlocked"]:
            # total runtime = iterations per item * sleep time for spellslot creation
            total_runtime += data["target"] * ((SLEEP_DURATION/2 + .1) * 3 + 2)

    return total_runtime

def move_and_click(coord_x: int, coord_y: int):
    """Moves the mouse to an onscreen coordinate and clicks."""
    
    moveTo(coord_x, coord_y)
    sleep(SLEEP_DURATION/2) # small buffer
    
    mouseDown()
    sleep(0.1) # hold for 100ms
    mouseUp()

def select_metamagic(type: str):
    """Select the metamagic icons in the UI."""
    
    if type == "SORCPTS": # 1745 1215
        move_and_click(1745,1215)
    elif type == "SPELLSLOTS": # 1745 1275
        move_and_click(1745,1275)
    else: # error, literally should not happen
        print("Invalid spell slot level.")
        return

# level 1:
# - 1305 1250
# level 2:
# - 1275 1250
# - 1335 1250
# level 3:
# - 1250 1250
# - 1305 1250
# - 1365 1250
# level 4:
# - 1215 1250
# - 1275 1250
# - 1335 1250
# - 1395 1250
# level 5:
# - 1190 1250
# - 1250 1250
# - 1305 1250
# - 1365 1250
# - 1425 1250
# level 6:
# - 1160 1250
# - 1220 1250
# - 1280 1250
# - 1335 1250
# - 1395 1250
# - 1450 1250

def create_sorcery_pts(spellslot_level: int):
    """function def."""
    
    spellslot_level_y = 1250
    spellslot_level_x = 1305
    if unlocked_spellslots_2:
        match (spellslot_level):
            case 1:
                spellslot_level_x = 1275
            case 2:
                spellslot_level_x = 1335
        
        if unlocked_spellslots_3:
            match (spellslot_level):
                case 1:
                    spellslot_level_x = 1250
                case 2:
                    spellslot_level_x = 1305
                case 3:
                    spellslot_level_x = 1365
            
            if unlocked_spellslots_4:
                match (spellslot_level):
                    case 1:
                        spellslot_level_x = 1215
                    case 2:
                        spellslot_level_x = 1275
                    case 3:
                        spellslot_level_x = 1335
                    case 4:
                        spellslot_level_x = 1395
                
                if unlocked_spellslots_5:
                    match (spellslot_level):
                        case 1:
                            spellslot_level_x = 1190
                        case 2:
                            spellslot_level_x = 1250
                        case 3:
                            spellslot_level_x = 1305
                        case 4:
                            spellslot_level_x = 1365
                        case 5:
                            spellslot_level_x = 1425
                    
                    if unlocked_spellslots_6:
                        match (spellslot_level):
                            case 1:
                                spellslot_level_x = 1160
                            case 2:
                                spellslot_level_x = 1220
                            case 3:
                                spellslot_level_x = 1280
                            case 4:
                                spellslot_level_x = 1335
                            case 5:
                                spellslot_level_x = 1395
                            case 6:
                                spellslot_level_x = 1450

    select_metamagic("SORCPTS")

    # select spell slot level
    move_and_click(spellslot_level_x,spellslot_level_y)
    
    # cast (mouse pos 1200 1050 minimum)
    move_and_click(1200,1050)
    
    sleep(2)

def create_spellslot(spellslot_level: int):
    """function def."""
    
    spellslot_level_y = 1250
    spellslot_level_x = 1305
    if unlocked_spellslots_2:
        match (spellslot_level):
            case 1:
                spellslot_level_x = 1275
            case 2:
                spellslot_level_x = 1335
        
        if unlocked_spellslots_3:
            match (spellslot_level):
                case 1:
                    spellslot_level_x = 1250
                case 2:
                    spellslot_level_x = 1305
                case 3:
                    spellslot_level_x = 1365
            
            if unlocked_spellslots_4:
                match (spellslot_level):
                    case 1:
                        spellslot_level_x = 1215
                    case 2:
                        spellslot_level_x = 1275
                    case 3:
                        spellslot_level_x = 1335
                    case 4:
                        spellslot_level_x = 1395
                
                if unlocked_spellslots_5:
                    match (spellslot_level):
                        case 1:
                            spellslot_level_x = 1190
                        case 2:
                            spellslot_level_x = 1250
                        case 3:
                            spellslot_level_x = 1305
                        case 4:
                            spellslot_level_x = 1365
                        case 5:
                            spellslot_level_x = 1425
                    
                    if unlocked_spellslots_6:
                        match (spellslot_level):
                            case 1:
                                spellslot_level_x = 1160
                            case 2:
                                spellslot_level_x = 1220
                            case 3:
                                spellslot_level_x = 1280
                            case 4:
                                spellslot_level_x = 1335
                            case 5:
                                spellslot_level_x = 1395
                            case 6:
                                spellslot_level_x = 1450
    
    # goto spellslot icon (mouse pos 1745 1275)
    select_metamagic("SPELLSLOTS")
    
    # goto spellslot level (dynamic mouse pos)
    move_and_click(spellslot_level_x, spellslot_level_y)
    
    # cast (mouse pos 1200 1050 minimum)
    move_and_click(1200,950)
    
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
        create_sorcery_pts(2)
    
    #print("Unequipping item")
    move_and_click(1285,1330)
    
    sleep(.025)
    

def main():
    print(f"Press '{PAUSE_HOTKEY}' to pause/resume or '{EXIT_HOTKEY}' to quit.")
    
    # Bind the macro to a hotkey (e.g., CTRL+ALT+M)
    try:
        add_hotkey('shift+r', macro)
        add_hotkey('esc', wait, args='esc')
    except KeyboardInterrupt:
        print("Macro stopped")
    
    # Keep the script running to listen for the hotkey
    #wait('esc')  # Exit the script by pressing ESC


if __name__ == "__main__":
    main()

