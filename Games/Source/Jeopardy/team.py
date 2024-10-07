
# 
# 
# 

from debug_logger import DebugLogger

class Team:
    def __init__(self, name: str) -> None:
        self.name = name
        self.score = 0
        self.logger = DebugLogger()

    # pre-condition: points must be initialized to a non-negative integer
    # post-condition: adds the specified points to the team's score
    def updateScore(self, points: int) -> None:
        """Update the team's score by the given points."""
        
        self.score += points
        self.logger.log(f"Increased this team's points by {points}",for_debug=False)
    
    # pre-condition: 
    # post-condition: 
    def resetTeam(self, full_reset: bool = False) -> None:
        self.score = 0
        if full_reset:
            self.name = ""

    def __str__(self) -> str:
        return f"{self.name} ({self.score} points)"
