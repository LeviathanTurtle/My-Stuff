
# 
# William Wadsworth
# 1.23.2025
# 
# 1.0 - initial release
# 1.01 - added amulet functionality
# 1.02 - fixed mouse coordinates based on unlocked spell slot levels
# 1.1 - dynamic data gen for mod support, pausing, estimated runtime, reduction in sleep time
# 1.2 - freecast, runtime args
# 1.3 - moved values to separate config file
# 
# Known issues:
# - exit and pause hotkeys are not working
# - estimated runtime is wrong
# 
# Possible planned updates:
# - get current spellslot data from game via a mod (requires research)
# - update icon coords to be based on UI levels (stack) (2-4) via extra logic
# - note 1:54:00 in that long video for optimization (time cut)
# 

from argparse import ArgumentParser
from typing import Tuple, Optional, Literal
from keyboard import add_hotkey, wait
from pydirectinput import moveTo, mouseDown, mouseUp
from dotenv import load_dotenv
from time import sleep, time

import threading
import sys
import os

import config
from config import PATCH_VERSION, COORDINATE_MAP, CoordType

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))
from Tools.Source.progress_bar import ProgressBar


# the length of time to sleep between inputs (in seconds)
SLEEP_DURATION: float = .025
MACRO_START_TIME = 0

is_paused = False
macro_lock = threading.Lock()

spell_costs = [2,3,5,6,7]
spellslot_data = {
    level: {
        "target": getattr(config, f"Target_spellslots_{level}"),
        "current": getattr(config, f"Current_spellslots_{level}"),
        "needed": max(0, getattr(config, f"Target_spellslots_{level}")-getattr(config, f"Current_spellslots_{level}")),
        "unlocked": True if config.MAX_SPELL_LEVEL>=level else False,
        "cost": spell_costs[level-1]
    }
    for level in range(1, config.MAX_SPELL_LEVEL+1)
}
Needed_sorc_pts = max(0, config.Target_sorc_pts-config.Current_sorc_pts)
macro_lock = threading.Lock()


def macro(
    using_shield: bool,
    using_amulet: bool,
    using_freecast: bool
) -> None:
    """Runs the macro, starting with creating the sorcery points from the equipment then creating
    the spell slots."""
    
    global MACRO_START_TIME
    MACRO_START_TIME = time()
    LOOP_COUNTER: int = 0
    
    # -------------------------------------------
    
    if not using_freecast:
        LOOP_COUNTER = update_globals(using_shield,using_amulet,using_freecast)
    else:
        update_globals(using_shield,using_amulet,using_freecast)
    
    #print(spellslot_data)
    #sys.exit()
    
    if not using_freecast:
        print(f"Need {LOOP_COUNTER} more sorc pts ({LOOP_COUNTER+config.Current_sorc_pts} total)")
        
        progress = ProgressBar(LOOP_COUNTER)

        # create sorcery points
        for i in range(LOOP_COUNTER):
            wait_if_paused()
            use_equipment(using_shield) if using_shield else use_equipment(using_amulet)
            progress.update(i+1)
        
        # loop through levels and create spellslots if unlocked
        for level, data in spellslot_data.items():
            if data["unlocked"]: # check if the level is unlocked
                print(f"Creating {data['needed']} lvl {level} spellslots...")
                progress = ProgressBar(data["needed"])
                iteration_cnt = 0
                
                for _ in range(data["needed"]):
                    wait_if_paused()
                    spend_stuff("SPELLSLOTS",level)
                    
                    iteration_cnt += 1
                    progress.update(iteration_cnt)
    else:
        freecast()
    
    print(f"Macro completed in {time()-MACRO_START_TIME:.2f}s")
    sys.exit()

def macro_threaded(
    using_shield: bool,
    using_amulet: bool,
    using_freecast: bool
) -> None:
    """function def."""
    
    if macro_lock.locked():
        print("Macro already running. Please wait.")
        return
    with macro_lock:
        macro(using_shield, using_amulet, using_freecast)

# -------------------------------------------------------------------------------------------------

def update_globals(
    using_shield: bool,
    using_amulet: bool,
    using_freecast: bool
) -> Optional[int]:
    """Updates global variables to be used in macro."""
    
    #global spellslot_data, spell_costs, Needed_sorc_pts
    global spellslot_data, Needed_sorc_pts

    if using_shield:
        spellslot_data[1]['current'] = 0
        spellslot_data[1]['needed'] = spellslot_data[1]['target']
    if using_amulet:
        spellslot_data[2]['current'] = 0
        spellslot_data[2]['needed'] = spellslot_data[2]['target']
    
    loop_counter: int = 0
    if not using_freecast:
        #loop_counter = sum(slot['needed']*spell_costs[level] for level, slot in spellslot_data.items() if level == 1 or slot['unlocked'] == True)
        loop_counter = sum(slot['needed']*slot['cost'] for level, slot in spellslot_data.items() if level == 1 or slot['unlocked'] == True)
        loop_counter += Needed_sorc_pts

        if using_amulet:
            loop_counter = loop_counter // 2 if loop_counter%2 == 0 else (loop_counter+1) // 2
    
    # construct update message
    msg = ""
    if not using_freecast:
        msg += f"Generating {loop_counter} sorc pts ({config.Target_sorc_pts} pt targ"
        for level, slot in spellslot_data.items():
            if level == 1 or slot['unlocked'] == True:
                msg += f" + {slot['needed']*slot['cost']} for lvl {level} spellslots"
        msg += ")"
    else:
        msg += f"Generating {Needed_sorc_pts} sorc pts"
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
    
    #global spellslot_data, spell_costs, Needed_sorc_pts
    global spellslot_data, Needed_sorc_pts
    
    # determine needed values
    List_spell_slots = [
        spellslot_data[level]['needed'] 
        for level in range(1, config.MAX_SPELL_LEVEL+1)
        if spellslot_data[level]['unlocked']
    ]
    
    #print(f"{Target_sorc_pts}, {Current_sorc_pts}, {Needed_sorc_pts}")
    List_sorc_pts = find_combination(Needed_sorc_pts)
    print(f"Creating sorcery points from the following levels: {List_sorc_pts}")
    progress = ProgressBar(len(List_sorc_pts))
    
    # maximize gains for sorc pts
    iteration_cnt = 0
    for num in List_sorc_pts:
        print(f"Creating {num} sorcery points from spell level {num}...")
        toggle_freecast()
        spend_stuff("SORCPTS",num)
        
        iteration_cnt += 1
        progress.update(iteration_cnt)
    
    # maximize gains for ascending spellslots
    for level, needed_slots in enumerate(List_spell_slots,start=1):
        print(f"Creating {needed_slots} spell slots for spell level {level}")
        progress = ProgressBar(needed_slots)
        iteration_cnt = 0
        
        for _ in range(needed_slots):
            toggle_freecast()
            spend_stuff("SPELLSLOTS",level)
            
            iteration_cnt += 1
            progress.update(iteration_cnt)

def toggle_freecast() -> None:
    """Selects the equipment and freecast icon."""
    
    clothing_coords = COORDINATE_MAP[CoordType.CLOTHING]
    freecast_coords = COORDINATE_MAP[CoordType.FREECAST]
    
    # move mouse to equipment and click
    move_and_click(clothing_coords.x,clothing_coords.y)
    
    # move mouse to freecast icon and click
    move_and_click(freecast_coords.x,freecast_coords.y)

def find_combination(x):
    """Calculates the most efficient (least amount) combination of spell slot levels to create
    sorcery points."""
    
    spellslot_levels = [config.MAX_SPELL_LEVEL-i for i in range(config.MAX_SPELL_LEVEL)]
    result = []
    
    for num in spellslot_levels:
        while x >= num:
            result.append(num)
            x -= num
    
    return result

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
        move_and_click(COORDINATE_MAP[CoordType.SORC_PTS].x,COORDINATE_MAP[CoordType.SORC_PTS].y)
    elif type == "SPELLSLOTS": # 1745 1275
        move_and_click(COORDINATE_MAP[CoordType.SPELL_SLOTS].x,COORDINATE_MAP[CoordType.SPELL_SLOTS].y)
    else: # error, literally should not happen
        print("Invalid spell slot level.")
        return

def spend_stuff(
    select_icon: str = Literal["SORCPTS","SPELLSLOTS"],
    spellslot_level: int = Literal[1,2,3,4,5]
) -> None:
    """Consumes a specified spell level to create sorcery points ."""
    
    spellslot_level_y = COORDINATE_MAP[CoordType.SPELL_LEVEL_1_max2].y
    spellslot_level_x = COORDINATE_MAP[CoordType.SPELL_LEVEL_1_max2].x
    # we can assume level 2 because we do not need to select the spell level if we only have one
    # level unlocked
    
    # get the position of the FIRST ICON
    match (config.MAX_SPELL_LEVEL):
        case 3: spellslot_level_x = COORDINATE_MAP[CoordType.SPELL_LEVEL_1_max3].x
        case 4: spellslot_level_x = COORDINATE_MAP[CoordType.SPELL_LEVEL_1_max4].x
        case 5: spellslot_level_x = COORDINATE_MAP[CoordType.SPELL_LEVEL_1_max5].x
    
    # adjust coords to find intended icon based on unlocked spell levels
    if spellslot_level >= 2:
        spellslot_level_x += 60*(spellslot_level-1) # each icon is ~60 pixels
    
    if select_icon in ["SORCPTS","SPELLSLOTS"]:
        select_metamagic(select_icon)
    else: raise ValueError(f"Invalid select icon '{select_icon}'")

    # select spell slot level
    # if all we have is level 1 we can skip selecting metamagic icon
    if config.MAX_SPELL_LEVEL != 1:
        move_and_click(spellslot_level_x,spellslot_level_y)
    
    # cast (mouse pos (775-1945) 1050 minimum)
    move_and_click(COORDINATE_MAP[CoordType.CAST].x,COORDINATE_MAP[CoordType.CAST].y)
    
    sleep(1.9)

def use_equipment(using_shield: bool) -> None:
    """Equips, uses the equipment's spell slot to create sorc pts, then unequips it."""
    
    equipment1_coords = COORDINATE_MAP[CoordType.EQUIPMENT_1]
    equipment2_coords = COORDINATE_MAP[CoordType.EQUIPMENT_2]
    
    # put on the first shield
    move_and_click(equipment1_coords.x,equipment1_coords.y)
    
    # sorc pts based on the equipment
    if using_shield:
        spend_stuff("SORCPTS",1)
    else:
        spend_stuff("SORCPTS",2)
    
    # take off the equipment
    if PATCH_VERSION == 8:
        # use the second shield here
        move_and_click(equipment2_coords.x,equipment2_coords.y)
    else:
        move_and_click(equipment1_coords.x,equipment1_coords.y)
    
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
    
    return args.shield, args.amulet, args.freecast



def main() -> None:
    # check that the user has run the test_coords script
    load_dotenv(os.path.join(".env"))
    
    if os.getenv("CHECKED_MOUSE_COORDS", "") != "True":
        print("You did not check the mouse coordinates!")
        sys.exit(1)
    else: print("Macro started")
    sleep(5)
    
    # python3 bg3_inf_sorcspell.py <-shield | -amulet | -freecast>
    using_shield, using_amulet, using_freecast = handle_args()
    
    print(f"Press '{config.PAUSE_HOTKEY}' to pause/resume or '{config.EXIT_HOTKEY}' to quit.")
    
    def start_hotkey_listener():
        # add hotkeys
        add_hotkey(config.START_HOTKEY, macro_threaded, args=(using_shield, using_amulet, using_freecast))
        add_hotkey(config.PAUSE_HOTKEY, toggle_pause)
        add_hotkey(config.EXIT_HOTKEY, lambda: sys.exit(0))
        wait() # Keep this thread alive to listen

    listener_thread = threading.Thread(target=start_hotkey_listener, daemon=True)
    listener_thread.start()
    
    try:
        while True:
            sleep(1) # keep the main thread alive
            # todo: does not exit when macro finishes
    except KeyboardInterrupt:
        global MACRO_START_TIME
        print("Macro interrupted.")
        print(f"Macro completed in {time()-MACRO_START_TIME:.2f}s")


if __name__ == "__main__":
    main()

