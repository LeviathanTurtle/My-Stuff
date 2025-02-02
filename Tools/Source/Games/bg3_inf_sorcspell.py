
# 
# William Wadsworth
# 1.23.2025
# 
# 1.0 - initial release
# 1.01 - added amulet functionality
# 1.02 - fixed mouse coordinates based on unlocked spell slot levels
# 1.1 - dynamic data gen for mod support, pausing, estimated runtime, reduction in sleep time
# 
# Known issues:
# - escape and pause hotkeys not working
# - estimated runtime is wrong
# 

from keyboard import add_hotkey, wait
from pydirectinput import moveTo, mouseDown, mouseUp
from time import sleep, time

#################################################
# CHANGE ONLY THESE VALUES HERE

PAUSE_HOTKEY: str = 'p'
EXIT_HOTKEY: str = 'esc'
MAX_SPELL_LEVEL: int = 5 # note that this should be the max spell level we can convert from sorc pts

# set any of these to true if they are unlocked and you want them expanded
unlocked_spellslots_2: bool = True # this is only used for UI coords
unlocked_spellslots_3: bool = True
unlocked_spellslots_4: bool = True 
unlocked_spellslots_5: bool = True
unlocked_spellslots_6: bool = False # this is only used for UI coords
# set this to false if using the amulet
using_shield: bool = True

# UPDATE THESE VALUES TO REFLECT YOUR CURRENT SPELL SLOTS
# Note that the current number of spellslots for level 1 should be 0 if using the shield, likewise
# for level 2 if using the amulet
if using_shield:
    Current_spellslots_1: int = 0 # leave this 0
    Current_spellslots_2: int = 18
else:
    Current_spellslots_1: int = 1
    Current_spellslots_2: int = 0 # leave this 0
Current_spellslots_3: int = 15
Current_spellslots_4: int = 10
Current_spellslots_5: int = 20
Current_sorc_pts: int = 198

# THESE ARE THE TARGET VALUES YOU WANT
Target_spellslots_1: int = 20
Target_spellslots_2: int = 20
Target_spellslots_3: int = 15
Target_spellslots_4: int = 10
Target_spellslots_5: int = 20
Target_sorc_pts: int = 30

#################################################

# the length of time to sleep between inputs (in seconds)
SLEEP_DURATION: float = .05
is_paused = False

def macro() -> None:
    """function def."""

    global Target_spellslots_1, Target_spellslots_2, Target_spellslots_3, Target_spellslots_4, Target_spellslots_5, Target_sorc_pts

    # SETUP:
    # LEVEL -> needed sorc pts per level:
    # I -> [x2]
    Target_spellslots_1 = max(0, Target_spellslots_1-Current_spellslots_1)
    Spell_1_pts = Target_spellslots_1*2

    # II -> [x3]
    Target_spellslots_2 = max(0, Target_spellslots_2-Current_spellslots_2)
    Spell_2_pts = Target_spellslots_2*3
    
    Target_sorc_pts = max(0, Target_sorc_pts-Current_sorc_pts)

    # the loop counter needs to be the total amount of sorcery points needed
    LOOP_COUNTER: int = Spell_1_pts+Spell_2_pts + Target_sorc_pts
    # III -> [x5]
    if unlocked_spellslots_3:
        Target_spellslots_3 = max(0, Target_spellslots_3-Current_spellslots_3)
        Spell_3_pts = Target_spellslots_3*5
        LOOP_COUNTER += Spell_3_pts
        
        # IV -> [x6]
        if unlocked_spellslots_4:
            Target_spellslots_4 = max(0, Target_spellslots_4-Current_spellslots_4)
            Spell_4_pts = Target_spellslots_4*6
            LOOP_COUNTER += Spell_4_pts
            
            # V -> [x7]
            if unlocked_spellslots_5:
                Target_spellslots_5 = max(0, Target_spellslots_5-Current_spellslots_5)
                Spell_5_pts = Target_spellslots_5*7
                LOOP_COUNTER += Spell_5_pts

    # if we are using the amulet, update the loop counter to use half the required iterations
    if not using_shield:
        # if it is odd, offset by 1 so we are over requirement
        LOOP_COUNTER = LOOP_COUNTER // 2 + (LOOP_COUNTER % 2)

    # define dict to be used in main loop
    #: dict[int, dict[str, any]]
    spellslot_data = {
        level: {
            "target": max(0, globals()[f"Target_spellslots_{level}"]),
            "unlocked": globals().get(f"unlocked_spellslots_{level}", True)
        }
        for level in range(1, MAX_SPELL_LEVEL+1)
    }
       
    print(f"Need {LOOP_COUNTER} more sorc pts ({LOOP_COUNTER+Current_sorc_pts} total)")
    est_runtime: float = ((SLEEP_DURATION/2 + .01) * 5 + 1.925) * LOOP_COUNTER + estimate_runtime(spellslot_data)
    print(f"Estimated runtime: {est_runtime:.2f}s ({est_runtime/60:.2f} min)")
    
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
    print(msg+")...")
    
    for _ in range(LOOP_COUNTER):
        activate_equipment(using_shield)

    # loop through levels and create spellslots if unlocked
    for level, data in spellslot_data.items():
        if data["unlocked"]:  # check if the level is unlocked
            print(f"Creating {data['target']} lvl {level} spellslots...")
            for _ in range(data["target"]):
                create_spellslot(level)
    
    print(f"Macro completed in {time()-start_time:.2f}s")

# -------------------------------------------------------------------------------------------------

def wait_if_paused() -> None:
    """Pauses the script if the pause hotkey is pressed."""
    
    while is_paused:
        sleep(1)

def toggle_pause() -> None:
    """function def."""
    
    global is_paused
    
    is_paused = not is_paused
    print(f"Macro {'paused' if is_paused else 'resumed'}.")

# 
def estimate_runtime(spellslot_data) -> float:
    """function def."""

    return sum(data["target"] * ((SLEEP_DURATION/2 + .01) * 3 + 1.9) for data in spellslot_data.values() if data["unlocked"])

def move_and_click(coord_x: int, coord_y: int) -> None:
    """Moves the mouse to an onscreen coordinate and clicks."""
    
    moveTo(coord_x, coord_y)
    sleep(SLEEP_DURATION/2) # small buffer
    
    mouseDown()
    sleep(0.01) # hold for 10ms
    mouseUp()

def select_metamagic(type: str) -> None:
    """Select the metamagic icons in the UI."""
    
    if type == "SORCPTS": # 1745 1215
        move_and_click(1745,1215)
    elif type == "SPELLSLOTS": # 1745 1275
        move_and_click(1745,1275)
    else: # error, literally should not happen
        print("Invalid spell slot level.")
        return

def create_sorcery_pts(spellslot_level: int) -> None:
    """Consumes a specified spell level to create sorcery points."""
    
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
                                spellslot_level_x = 1340
                            case 5:
                                spellslot_level_x = 1400
                            # we are ignoring lvl 6 because we cannot create that spellslot

    select_metamagic("SORCPTS")

    # select spell slot level
    move_and_click(spellslot_level_x,spellslot_level_y)
    
    # cast (mouse pos 1200 1050 minimum)
    move_and_click(1200,1050)
    
    sleep(1.9)

def create_spellslot(spellslot_level: int) -> None:
    """Consumes sorcery points to create a spell slot at the specified level."""
    
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
    
    # goto spellslot icon (mouse pos 1745 1275)
    select_metamagic("SPELLSLOTS")
    
    # goto spellslot level (dynamic mouse pos)
    move_and_click(spellslot_level_x, spellslot_level_y)
    
    # cast (mouse pos 1200 1050 minimum)
    move_and_click(1200,950)
    
    sleep(1.9)

def activate_equipment(using_shield: bool) -> None:
    """Equips and unequips the equipment for the exploit."""
    
    # --- EVENT LOOP: ---
    #   move mouse (1285 1330)
    #   click (equip item)
    #   create_sorc_pts
    #   move mouse
    #   click (unequip item)
    
    #print("Equipping item")
    move_and_click(1285,1330)
    
    # sorc pts based on the equipment
    if using_shield:
        create_sorcery_pts(1)
    else:
        create_sorcery_pts(2)
    
    #print("Unequipping item")
    move_and_click(1285,1330)
    
    sleep(.025)
    

def main() -> None:
    print(f"Press '{PAUSE_HOTKEY}' to pause/resume or '{EXIT_HOTKEY}' to quit.")
    
    try:
        # add hotkeys
        add_hotkey('shift+r', macro)
        add_hotkey(PAUSE_HOTKEY, toggle_pause)
        add_hotkey(EXIT_HOTKEY, exit)
        
        # keep the script running to listen for the hotkey
        wait()
    except KeyboardInterrupt:
        print("Macro stopped.")


if __name__ == "__main__":
    main()

