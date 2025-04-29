
# 
# William Wadsworth
# 

from dataclasses import dataclass
from enum import Enum

#################################################
######## THSE ARE THE MAIN MACRO VALUES #########
#################################################

START_HOTKEY: str = 'shift+r'
PAUSE_HOTKEY: str = 'p'
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
#################################################



class CoordType(Enum):
    EQUIPMENT = "equipment"
    FREECAST = "freecast"
    CLOTHING = "clothing"
    SORC_PTS = "sorcery_pts"
    SPELL_SLOTS = "spell_slots"
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
    CoordType.EQUIPMENT: Coordinates(x=1285, y=1330),
    CoordType.FREECAST: Coordinates(x=1660, y=1160),
    CoordType.CLOTHING: Coordinates(x=1660, y=1220),
    CoordType.SORC_PTS: Coordinates(x=1745, y=1215),
    CoordType.SPELL_SLOTS: Coordinates(x=1745, y=1275),
    
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