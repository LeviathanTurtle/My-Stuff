
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
    EQUIPMENT = "equipment"
    FREECAST = "freecast"
    CLOTHING = "clothing"
    SORC_PTS = "sorcery_pts"
    SPELL_SLOTS = "spell_slots"
    CAST = "cast"
    #SPELL_LEVEL_2_max2 = "spell_level_2_max2"
    #
    #SPELL_LEVEL_2_max3 = "spell_level_2_max3"
    #SPELL_LEVEL_3_max3 = "spell_level_3_max3"
    #
    #SPELL_LEVEL_2_max4 = "spell_level_2_max4"
    #SPELL_LEVEL_3_max4 = "spell_level_3_max4"
    #SPELL_LEVEL_4_max4 = "spell_level_4_max4"
    #
    #SPELL_LEVEL_2_max5 = "spell_level_2_max5"
    #SPELL_LEVEL_3_max5 = "spell_level_3_max5"
    #SPELL_LEVEL_4_max5 = "spell_level_4_max5"
    #SPELL_LEVEL_5_max5 = "spell_level_5_max5"

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
    CoordType.EQUIPMENT: Coordinates(x=1910, y=1220),
    
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
    
    # each spell level
    # max level 2
    #CoordType.SPELL_LEVEL_2_max2: Coordinates(x=1275, y=1250),
    
    # max level 3
    #CoordType.SPELL_LEVEL_2_max3: Coordinates(x=1250, y=1250),
    #CoordType.SPELL_LEVEL_3_max3: Coordinates(x=0, y=1250),
    
    # max level 4
    #CoordType.SPELL_LEVEL_2_max4: Coordinates(x=1220, y=1250),
    #
    #
    
    # max level 5
    #CoordType.SPELL_LEVEL_2_max5: Coordinates(x=1190, y=1250),
    #
    #
    #
}

#################################################
#################################################