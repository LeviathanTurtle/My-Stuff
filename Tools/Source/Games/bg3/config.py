
# 
# William Wadsworth
# 

from dataclasses import dataclass
from enum import Enum

#################################################
######## THESE ARE THE MAIN MACRO VALUES ########
#################################################

START_HOTKEY: str = 'shift+r'
PAUSE_HOTKEY: str = 'p'
EXIT_HOTKEY: str = 'esc'

# NOTE: this value should be the max spell level you can convert from sorc pts
# Vanilla: [1,5]
MAX_SPELL_LEVEL: int = 4

# UPDATE THESE VALUES TO REFLECT YOUR CURRENT SPELL SLOTS
Current_spellslots_1: int = 3
Current_spellslots_2: int = 2
Current_spellslots_3: int = 3
Current_spellslots_4: int = 0
Current_spellslots_5: int = 1
Current_sorc_pts: int = 4

# THESE ARE THE TARGET VALUES YOU WANT
Target_spellslots_1: int = 10
Target_spellslots_2: int = 5
Target_spellslots_3: int = 5
Target_spellslots_4: int = 5
Target_spellslots_5: int = 50
Target_sorc_pts: int = 30

#################################################
#################################################



class CoordType(Enum):
    EQUIPMENT_1 = "shield/amulet slot"
    EQUIPMENT_2 = "second shield slot (patch 8)"
    FREECAST = "freecast icon"
    CLOTHING = "clothing icon"
    SORC_PTS = "sorcery pts icon"
    SPELL_SLOTS = "spell slots icon"
    CAST = "cast spot"
    
    SPELL_LEVEL_1_max2 = "first spell level with a max spell level of 2"
    SPELL_LEVEL_1_max3 = "first spell level with a max spell level of 3"
    SPELL_LEVEL_1_max4 = "first spell level with a max spell level of 4"
    SPELL_LEVEL_1_max5 = "first spell level with a max spell level of 5"

@dataclass
class Coordinates:
    x: int
    y: int

#################################################
############### MOUSE COORDINATES ###############
#################################################

#EQUIPMENT_COORDS_X: int = 1285
#EQUIPMENT_COORDS_Y: int = 1330
#EQUIPMENT_COORDS = Coordinates(x=1285, y=1330)

COORDINATE_MAP = {
    # Coordinates of the shield or amulet
    # Default: 1285, 1330
    CoordType.EQUIPMENT_1: Coordinates(x=1910, y=1220),
    
    # Coordinates of the second shield (for Patch 8)
    # Default: 
    CoordType.EQUIPMENT_2: Coordinates(x=0, y=0),
    
    # Coordinates of freecast icon
    # Default: 1660, 1160
    CoordType.FREECAST: Coordinates(x=1660, y=1160),
    
    # Coordinates of clothing icon (used with freecast)
    # Default: 1660, 1220
    CoordType.CLOTHING: Coordinates(x=1660, y=1220),
    
    # Coordinates of sorcery point conversion metamagic icon
    # Default: 1745, 1215
    CoordType.SORC_PTS: Coordinates(x=1850, y=1280),
    
    # Coordinates of spell slot conversion metamagic icon
    # Default: 1745, 1275
    CoordType.SPELL_SLOTS: Coordinates(x=1850, y=1340),
    
    # Coordinates of the spot above the UI to cast the thing
    # Default: 1200, 950
    CoordType.CAST: Coordinates(x=1200, y=950),
    
    # Coordinates of the fist spell level spot with a max spell level of 2
    # Default: 
    CoordType.SPELL_LEVEL_1_max2: Coordinates(x=1275, y=1250),
    
    # Coordinates of the fist spell level spot with a max spell level of 3
    # Default: 
    CoordType.SPELL_LEVEL_1_max3: Coordinates(x=1250, y=1250),
    
    # Coordinates of the fist spell level spot with a max spell level of 4
    # Default: 
    CoordType.SPELL_LEVEL_1_max4: Coordinates(x=1220, y=1250),
    
    # Coordinates of the fist spell level spot with a max spell level of 5
    # Default: 
    CoordType.SPELL_LEVEL_1_max5: Coordinates(x=1190, y=1250),
}

#################################################
#################################################