

from enum import Enum


class ElevatorStatus(Enum):
    """Enumerated type. Defines the 6 unique possible states of an elevator"""
    AVAILABLE = 0
    IN_MAINTENANCE = 1
    ASCENDING = 2
    DESCENDING = 3
    ARRIVING_AT_FLOOR = 4
    MOVING_DOORS = 5


class Elevator:
    """Model class. A generic representation of an elevator."""
    def __init__(self, id: int, weight_limit: float, current_floor: int, travel_speed: float, 
                 status: ElevatorStatus, hours_operated: float, hours_since_last_maint: float):
        self.id =  id
        self.weight_limit = weight_limit
        self.current_floor = current_floor
        self.travel_speed = travel_speed
        self.wait_timer = WAIT_TIMERS.get(status,0)
        self.hours_operated = hours_operated
        self.hours_since_last_maint = hours_since_last_maint
        
    def ascend(self):
        pass
    
    def descend(self):
        pass
    
    def check_for_maintenance(self) -> bool:
        if(self.hours_since_last_maint >= 200):
            return True
        else:
            return False


WAIT_TIMERS: dict[ElevatorStatus,int] = {
    ElevatorStatus.ASCENDING : 10, # seconds
    ElevatorStatus.DESCENDING : 10, # seconds
    ElevatorStatus.ARRIVING_AT_FLOOR : 5, # seconds
    ElevatorStatus.MOVING_DOORS: 3 # seconds
}