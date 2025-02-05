
# 
# William Wadsworth
# 1.23.2025
# 
# 1.0 - initial release
# 1.01 - added amulet functionality
# 1.02 - fixed mouse coordinates based on unlocked spell slot levels
# 1.1 - dynamic data gen for mod support, pausing, estimated runtime, reduction in sleep time
# 1.2 - freecast, runtime args
# 
# Known issues:
# - escape and pause hotkeys not working
# - estimated runtime is wrong
# 
# Possible planned updates:
# - get current spellslot data from game via a mod
# 

from argparse import ArgumentParser
from typing import Tuple, Optional
from keyboard import add_hotkey, wait
from pydirectinput import moveTo, mouseDown, mouseUp
from time import sleep, time, perf_counter

#################################################
# CHANGE ONLY THESE VALUES HERE

PAUSE_HOTKEY: str = 'p'
EXIT_HOTKEY: str = 'esc'
MAX_SPELL_LEVEL: int = 5 # NOTE: this should be the max spell level you can convert from sorc pts

# set any of these to true if they are unlocked and you want them expanded
unlocked_spellslots_2: bool = True # this is only used for UI coords
unlocked_spellslots_3: bool = True
unlocked_spellslots_4: bool = True 
unlocked_spellslots_5: bool = True
unlocked_spellslots_6: bool = False # this is only used for UI coords

# UPDATE THESE VALUES TO REFLECT YOUR CURRENT SPELL SLOTS
Current_spellslots_1: int = 4
Current_spellslots_2: int = 3
Current_spellslots_3: int = 3
Current_spellslots_4: int = 3
Current_spellslots_5: int = 3
Current_sorc_pts: int = 6

# THESE ARE THE TARGET VALUES YOU WANT
Target_spellslots_1: int = 15
Target_spellslots_2: int = 15
Target_spellslots_3: int = 15
Target_spellslots_4: int = 15
Target_spellslots_5: int = 50
Target_sorc_pts: int = 30

#################################################

# the length of time to sleep between inputs (in seconds)
SLEEP_DURATION: float = .05
is_paused = False
Needed_spellslots_1: int = 0
Needed_spellslots_2: int = 0
Needed_spellslots_3: int = 0
Needed_spellslots_4: int = 0
Needed_spellslots_5: int = 0

def macro(
    using_shield: bool,
    using_amulet: bool,
    using_freecast: bool
) -> None:
    """Runs the macro, starting with creating the sorcery points from the equipment then creating
    the spell slots."""

    print("Macro started")

    # define dict to be used in main loop
    #: dict[int, dict[str, any]]
    spellslot_data = {
        level: {
            "target": max(0, globals()[f"Target_spellslots_{level}"]),
            "unlocked": globals().get(f"unlocked_spellslots_{level}", True)
        }
        for level in range(1, MAX_SPELL_LEVEL+1)
    }
    
    sleep(5)
    start_time = time()
    
    # -------------------------------------------
    
    LOOP_COUNTER: int = 0
    if not using_freecast:
        LOOP_COUNTER = update_globals(using_shield,using_amulet,using_freecast)
    else:
        update_globals(using_shield,using_amulet,using_freecast)
    
    if not using_freecast:
        print(f"Need {LOOP_COUNTER} more sorc pts ({LOOP_COUNTER+Current_sorc_pts} total)")
        est_runtime: float = ((SLEEP_DURATION/2 + .01) * 5 + 1.95) * LOOP_COUNTER + estimate_runtime(spellslot_data)
        print(f"Estimated runtime: {est_runtime:.2f}s ({est_runtime/60:.2f} min)")
        
        for _ in range(LOOP_COUNTER):
            activate_equipment(using_shield) if using_shield else activate_equipment(using_amulet)

        # loop through levels and create spellslots if unlocked
        for level, data in spellslot_data.items():
            if data["unlocked"]: # check if the level is unlocked
                print(f"Creating {data['target']} lvl {level} spellslots...")
                for _ in range(data["target"]):
                    create_spellslot(level)
    else:
        freecast()
    
    print(f"Macro completed in {time()-start_time:.2f}s")

# -------------------------------------------------------------------------------------------------

def update_globals(
    using_shield: bool,
    using_amulet: bool,
    using_freecast: bool
) -> Optional[int]:
    """function def."""
    
    global Target_spellslots_1, Target_spellslots_2, Target_spellslots_3, Target_spellslots_4, Target_spellslots_5, Target_sorc_pts
    global Current_spellslots_1, Current_spellslots_2, Current_spellslots_3, Current_spellslots_4, Current_spellslots_5, Current_sorc_pts
    global Needed_spellslots_1, Needed_spellslots_2, Needed_spellslots_3, Needed_spellslots_4, Needed_spellslots_5, Needed_sorc_pts
    
    # SETUP:
    # LEVEL -> needed sorc pts per level:
    # I -> [x2]
    if using_shield: Current_spellslots_1 = 0
    Needed_spellslots_1 = max(0, Target_spellslots_1-Current_spellslots_1)

    # II -> [x3]
    if using_amulet: Current_spellslots_2 = 0
    Needed_spellslots_2 = max(0, Target_spellslots_2-Current_spellslots_2)
    
    Needed_sorc_pts = max(0, Target_sorc_pts-Current_sorc_pts)
    
    if not using_freecast:
        Spell_1_pts = Target_spellslots_1*2
        Spell_2_pts = Target_spellslots_2*3

        # the loop counter needs to be the total amount of sorcery points needed
        loop_counter: int = Spell_1_pts+Spell_2_pts + Target_sorc_pts
        
    # III -> [x5]
    if unlocked_spellslots_3:
        Needed_spellslots_3 = max(0, Target_spellslots_3-Current_spellslots_3)
        
        if not using_freecast:
            Spell_3_pts = Target_spellslots_3*5
            loop_counter += Spell_3_pts
        
        # IV -> [x6]
        if unlocked_spellslots_4:
            Needed_spellslots_4 = max(0, Target_spellslots_4-Current_spellslots_4)
            
            if not using_freecast:
                Spell_4_pts = Target_spellslots_4*6
                loop_counter += Spell_4_pts
            
            # V -> [x7]
            if unlocked_spellslots_5:
                Needed_spellslots_5 = max(0, Target_spellslots_5-Current_spellslots_5)
                
                if not using_freecast:
                    Spell_5_pts = Target_spellslots_5*7
                    loop_counter += Spell_5_pts

    # if we are using the amulet, update the loop counter to use half the required iterations
    if using_amulet:
        # if it is odd, offset by 1 so we are over requirement
        loop_counter = loop_counter // 2 + (loop_counter % 2)
    
    # get all necessary sorc pts
    msg: str = ""
    if not using_freecast:
        msg += f"Getting {loop_counter} sorc pts ({Target_sorc_pts} pt targ + {Spell_1_pts} for lvl 1 spellslots + {Spell_2_pts} for lvl 2 spellslots"
    else:
        msg += f"Getting {Needed_sorc_pts} sorc pts, {Needed_spellslots_1} lvl 1 spellslots, {Needed_spellslots_2} lvl 2 spellslots"
    
    if unlocked_spellslots_3:
        if not using_freecast:
            msg += f" + {Spell_3_pts} for lvl 3 spellslots"
        else:
            msg += f", {Needed_spellslots_3} for lvl 3 spellslots"
        
        if unlocked_spellslots_4:
            if not using_freecast:
                msg += f" + {Spell_4_pts} for lvl 4 spellslots"
            else:
                msg += f", {Needed_spellslots_4} for lvl 4 spellslots"
                
            if unlocked_spellslots_5:
                if not using_freecast:
                    msg += f" + {Spell_5_pts} for lvl 5 spellslots"
                else:
                    msg += f", {Needed_spellslots_5} for lvl 5 spellslots"
    if not using_freecast:
        msg += ")"
    
    print(msg+"...")
    
    if not using_freecast:
        return loop_counter
    else: return
    

def wait_if_paused() -> None:
    """Pauses the script if the pause hotkey is pressed."""
    
    while is_paused:
        sleep(1)

def toggle_pause() -> None:
    """Helper function to raise or lower the pause flag."""
    
    global is_paused
    
    is_paused = not is_paused
    print(f"Macro {'paused' if is_paused else 'resumed'}.")

def freecast() -> None:
    """function def."""
    
    global Target_spellslots_1, Target_spellslots_2, Target_spellslots_3, Target_spellslots_4, Target_spellslots_5, Target_sorc_pts
    global Current_spellslots_1, Current_spellslots_2, Current_spellslots_3, Current_spellslots_4, Current_spellslots_5, Current_sorc_pts
    global Needed_spellslots_1, Needed_spellslots_2, Needed_spellslots_3, Needed_spellslots_4, Needed_spellslots_5, Needed_sorc_pts
    
    # determine needed values
    List_spell_slots = [Needed_spellslots_1,Needed_spellslots_2]
    
    if unlocked_spellslots_3:
        List_spell_slots.append(Needed_spellslots_3)
        if unlocked_spellslots_4:
            List_spell_slots.append(Needed_spellslots_4)
            if unlocked_spellslots_5:
                List_spell_slots.append(Needed_spellslots_5)
    
    print(f"{Target_sorc_pts}, {Current_sorc_pts}, {Needed_sorc_pts}")
    
    # maximize gains for sorc pts
    List_sorc_pts = find_combination(Needed_sorc_pts)
    print(f"Creating sorcery points from the following levels: {List_sorc_pts}")
    for num in List_sorc_pts:
        print(f"Creating {num} sorcery points from spell level {num}")
        toggle_freecast()
        create_sorcery_pts(num)
    
    # maximize gains for ascending spellslots
    for index, value in enumerate(List_spell_slots,start=1):
        print(f"Creating {value} spell slots for spell level {index}")
        for _ in range(value):
            toggle_freecast()
            create_spellslot(index)

def toggle_freecast() -> None:
    """function def."""
    
    # move mouse to equipment and click
    move_and_click(1660,1220)
    
    # move mouse to freecast icon and click
    move_and_click(1660,1160)

def find_combination(x):
    """function def."""
    
    spellslot_levels = [5, 4, 3, 2, 1]
    result = []
    
    for num in spellslot_levels:
        while x >= num:
            result.append(num)
            x -= num
    
    return result

# 
def estimate_runtime(spellslot_data) -> float:
    """Estimate the main loop runtime of the macro."""
    
    start_time = perf_counter()
    time_elapsed: float = 0
    
    for level, data in spellslot_data.items():
        if data["unlocked"]: # check if the level is unlocked
            print(f"Creating {data['target']} lvl {level} spellslots...")
            for _ in range(data["target"]):
                time_elapsed += (SLEEP_DURATION/2 + .01) * 3 + 1.9

    return perf_counter()-start_time + time_elapsed

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
    # todo: change this to add ~60px per spell level unlocked
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
    
    sleep(.05)

def handle_args() -> Tuple[bool,bool,bool]:
    """Helper function to initiate runtime arguments."""
    
    # python3 bg3_inf_sorcspell.py <-shield | -amulet | -freecast>
    parser = ArgumentParser(description="Baldur's Gate 3 macro with optional flags.")
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-shield", action="store_true", help="Use the Shield of Devotion.")
    group.add_argument("-amulet", action="store_true", help="Use the Spell Savant Amulet.")
    group.add_argument("-freecast", action="store_true", help="Use the Illithid Freecast passive.")
    args = parser.parse_args()
    
    #using_shield: bool = args.shield
    #using_amulet: bool = args.amulet
    #using_freecast: bool = args.freecast
    return args.shield, args.amulet, args.freecast



def main() -> None:
    # python3 bg3_inf_sorcspell.py <-shield | -amulet | -freecast>
    using_shield, using_amulet, using_freecast = handle_args()
    
    print(f"Press '{PAUSE_HOTKEY}' to pause/resume or '{EXIT_HOTKEY}' to quit.")
    
    # try while true
    try:
        # add hotkeys
        add_hotkey('shift+r', macro, args=(using_shield,using_amulet,using_freecast))
        add_hotkey(PAUSE_HOTKEY, toggle_pause)
        add_hotkey(EXIT_HOTKEY, exit)
        
        # keep the script running to listen for the hotkey
        wait()
    except KeyboardInterrupt:
        print("Macro stopped.")


if __name__ == "__main__":
    main()

