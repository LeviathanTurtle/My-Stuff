# 
# 
# 

class Team:
    def __init__(self, name: str) -> None:
        """Represents a trivia team."""
        self.name = name
        self.score = 0

    # pre-condition: points must be initialized to a non-negative integer
    # post-condition: adds the specified points to the team's score
    def updateScore(self, points: int, debug: bool = False) -> None:
        """Update the team's score by the given points."""
        self.score += points
        if debug:
            print(f"[DEBUG] Increased this team's points by {points}")

    def __str__(self) -> str:
        return f"{self.name} ({self.score} points)"
