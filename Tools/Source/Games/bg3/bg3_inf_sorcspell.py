
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
# - exit and pause hotkeys are not working
# - estimated runtime is wrong
# 
# Possible planned updates:
# - get current spellslot data from game via a mod
# - HOLD ON note 1:54:00 in that long video for optimization (time cut)
# 

from argparse import ArgumentParser
from typing import Tuple, Optional, Literal
from keyboard import add_hotkey, wait
from pydirectinput import moveTo, mouseDown, mouseUp
from dotenv import load_dotenv
from os import getenv
from time import sleep, time, perf_counter
import sys

#################################################
# CHANGE ONLY THESE VALUES HERE

START_HOTKEY: str = 'shift+r'
PAUSE_HOTKEY: str = 'p' # todo: use env instead of bool?
EXIT_HOTKEY: str = 'esc'

# NOTE: this value should be the max spell level you can convert from sorc pts
# Vanilla: [1,5]
MAX_SPELL_LEVEL: int = 5

# UPDATE THESE VALUES TO REFLECT YOUR CURRENT SPELL SLOTS
Current_spellslots_1: int = 4
Current_spellslots_2: int = 3
Current_spellslots_3: int = 3
Current_spellslots_4: int = 3
Current_spellslots_5: int = 1
Current_sorc_pts: int = 6

# THESE ARE THE TARGET VALUES YOU WANT
Target_spellslots_1: int = 10
Target_spellslots_2: int = 10
Target_spellslots_3: int = 10
Target_spellslots_4: int = 10
Target_spellslots_5: int = 50
Target_sorc_pts: int = 60

#################################################

# the length of time to sleep between inputs (in seconds)
SLEEP_DURATION: float = .025
is_paused = False

spellslot_data = {
    level: {
        "target": globals()[f"Target_spellslots_{level}"],
        "current": globals()[f"Current_spellslots_{level}"],
        "needed": max(0, globals()[f"Target_spellslots_{level}"]-globals()[f"Current_spellslots_{level}"]),
        "unlocked": globals().get(f"unlocked_spellslots_{level}", True),
    }
    for level in range(1, MAX_SPELL_LEVEL+1)
}
Needed_sorc_pts = max(0, Target_sorc_pts-Current_sorc_pts)
spell_costs = [2,3,5,6,7]

def macro(
    using_shield: bool,
    using_amulet: bool,
    using_freecast: bool
) -> None:
    """Runs the macro, starting with creating the sorcery points from the equipment then creating
    the spell slots."""
    
    # check that the user has run the test_coords script
    load_dotenv()
    if getenv("CHECKED_MOUSE_COORDS", "").lower() != "True":
        print("You did not check the mouse coordinates!")
        sys.exit(1)
    else: print("Macro started")

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
        est_runtime: float = ((SLEEP_DURATION/2 + .01) * 5 + 1.95) * LOOP_COUNTER + estimate_runtime(using_freecast)
        print(f"Estimated runtime: {est_runtime:.2f}s ({est_runtime/60:.2f} min)")
        
        for _ in range(LOOP_COUNTER):
            activate_equipment(using_shield) if using_shield else activate_equipment(using_amulet)

        # loop through levels and create spellslots if unlocked
        for level, data in spellslot_data.items():
            if data["unlocked"]: # check if the level is unlocked
                print(f"Creating {data['target']} lvl {level} spellslots...")
                for _ in range(data["target"]):
                    spend_stuff("SPELLSLOTS",level)
    else:
        #est_runtime: float = ((SLEEP_DURATION/2 + .01) * 5 + 1.95) * LOOP_COUNTER + estimate_runtime(using_freecast)
        #print(f"Estimated runtime: {est_runtime:.2f}s ({est_runtime/60:.2f} min)")
        freecast()
    
    print(f"Macro completed in {time()-start_time:.2f}s")

# -------------------------------------------------------------------------------------------------

def update_globals(
    using_shield: bool,
    using_amulet: bool,
    using_freecast: bool
) -> Optional[int]:
    """Updates global variables to be used in macro."""
    
    global spellslot_data, spell_costs, Needed_sorc_pts

    if using_shield:
        spellslot_data[1]['current'] = 0
    if using_amulet:
        spellslot_data[2]['current'] = 0
    
    loop_counter: int = 0
    if not using_freecast:
        loop_counter = sum(slot['needed']*spell_costs[level] for level, slot in spellslot_data.items() if level == 1 or slot['unlocked'] == True)
        loop_counter += Needed_sorc_pts

        if using_amulet:
            loop_counter = loop_counter // 2 if loop_counter%2 == 0 else (loop_counter+1) // 2
    
    # construct update message
    msg = ""
    if not using_freecast:
        msg += f"Getting {loop_counter} sorc pts ({Target_sorc_pts} pt targ"
        for level, slot in spellslot_data.items():
            if level == 1 or slot['unlocked'] == True:
                msg += f" + {slot['needed']*slot['cost']} for lvl {level} spellslots"
        msg += ")"
    else:
        msg += f"Getting {Needed_sorc_pts} sorc pts"
        for level, slot in spellslot_data.items():
            if level == 1 or slot['unlocked'] == True:
                msg += f", {slot['needed']} lvl {level} spellslots"

    print(msg+"...")

    return loop_counter if not using_freecast else None

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
    """Uses the freecast method of creating spell slots and sorcery points."""
    
    global spellslot_data, spell_costs, Needed_sorc_pts
    
    # determine needed values
    List_spell_slots = [
        spellslot_data[level]['needed'] 
        for level in range(1, MAX_SPELL_LEVEL+1)
        if spellslot_data[level]['unlocked']
    ]
    
    #print(f"{Target_sorc_pts}, {Current_sorc_pts}, {Needed_sorc_pts}")
    List_sorc_pts = find_combination(Needed_sorc_pts)
    print(f"Creating sorcery points from the following levels: {List_sorc_pts}")
    
    # maximize gains for sorc pts
    for num in List_sorc_pts:
        print(f"Creating {num} sorcery points from spell level {num}")
        toggle_freecast()
        spend_stuff("SORCPTS",num)
    
    # maximize gains for ascending spellslots
    for level, needed_slots in enumerate(List_spell_slots,start=1):
        print(f"Creating {needed_slots} spell slots for spell level {level}")
        for _ in range(needed_slots):
            toggle_freecast()
            spend_stuff("SPELLSLOTS",level)

def toggle_freecast() -> None:
    """Selects the equipment and freecast icon."""
    
    # move mouse to equipment and click
    move_and_click(1660,1220)
    
    # move mouse to freecast icon and click
    move_and_click(1660,1160)

def find_combination(x):
    """Calculates the most efficient (least amount) combination of spell slot levels to create
    sorcery points."""
    
    spellslot_levels = [MAX_SPELL_LEVEL-i for i in range(MAX_SPELL_LEVEL)]
    result = []
    
    for num in spellslot_levels:
        while x >= num:
            result.append(num)
            x -= num
    
    return result

# 
def estimate_runtime(using_freecast: bool) -> float:
    """Estimate the main loop runtime of the macro."""
    
    global spellslot_data
    
    start_time = perf_counter()
    time_elapsed: float = 0
    
    if using_freecast:
        pass
    else:
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

def spend_stuff(
    select_icon: str = Literal["SORCPTS","SPELLSLOTS"],
    spellslot_level: int = Literal[1,2,3,4,5]
) -> None:
    """Consumes a specified spell level to create sorcery points ."""
    
    spellslot_level_y = 1250
    # note that this assumes level 2
    spellslot_level_x = 1275
    
    # get the position of the first icon
    match (MAX_SPELL_LEVEL):
        case 3:
            spellslot_level_x = 1250
        case 4:
            spellslot_level_x = 1220
        case 5:
            spellslot_level_x = 1190
    
    # adjust coords to find intended icon based on unlocked spell levels
    if spellslot_level >= 2:
        spellslot_level_x += 60*(spellslot_level-1)
    
    if select_icon in ["SORCPTS","SPELLSLOTS"]:
        select_metamagic(select_icon)
    else: raise ValueError(f"Invalid select icon: {select_icon}")

    # select spell slot level
    # if all we have is level 1 we can skip selecting metamagic icon
    if MAX_SPELL_LEVEL != 1:
        move_and_click(spellslot_level_x,spellslot_level_y)
    
    # cast (mouse pos (775-1945) 1050 minimum)
    move_and_click(1200,950)
    
    sleep(1.9)

def activate_equipment(using_shield: bool) -> None:
    """Equips and unequips the equipment for the exploit."""
    
    # put on the equipment
    move_and_click(1285,1330)
    
    # sorc pts based on the equipment
    if using_shield:
        spend_stuff("SORCPTS",1)
    else:
        spend_stuff("SORCPTS",2)
    
    # take off the equipment
    move_and_click(1285,1330)
    
    sleep(.05)

def handle_args() -> Tuple[bool,bool,bool]:
    """Helper function to initiate runtime arguments."""
    
    # todo: check env that user ran mouse test
    
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
    
    try:
        # add hotkeys
        add_hotkey(START_HOTKEY, macro, args=(using_shield,using_amulet,using_freecast))
        add_hotkey(PAUSE_HOTKEY, toggle_pause)
        add_hotkey(EXIT_HOTKEY, exit)
        
        # keep the script running to listen for the hotkey
        wait()
    except KeyboardInterrupt:
        print("Macro stopped.")


if __name__ == "__main__":
    main()

