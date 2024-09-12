# This uses the Trivia API from Open Trivia Database 
# (documentation: https://opentdb.com/api_config.php)
# 
# Updated 8.16.2024: function decomposition and PEP 8 Compliance
# Updated 8.24.2024: added the option to select a category, runtime flags
# 
# API notes are at the bottom.
# 
# USAGE:
# python3 trivia.py [-d] [-c] [-mc] <num_questions>
# Run 'python3 trivia.py' with "-h" or "--help" for argument descriptions

#from argparse import ArgumentParser
from requests import get
from html import unescape
from random import sample, shuffle
from typing import List, Optional, Union, Tuple, Dict

import tkinter as tk
from tkinter import messagebox, simpledialog


# pre-condition: an internet connection
# post-condition: if the API query is successful, it returns a dictionary containing the category
#                 IDs and names, otherwise None
def fetchCategories(
    debug: bool = False
) -> Optional[Dict[int, str]]:
    """Fetch trivia categories from the OpenTrivia Database API."""
    
    if debug:
        print("Entering fetchCategories...")
    
    response = get("https://opentdb.com/api_category.php")
    
    if response.status_code == 200:
        categories = response.json()['trivia_categories']
        return {cat['id']: cat['name'] for cat in categories}
    else:
        print(f"Error: {response.status_code}\nFailed to retrieve categories")
        if debug:
            print("Exiting fetchCategories.")
        return None


# pre-condition: num_questions is a positive integer, an internet connection
# post-condition: returns a list of question data if the API call is successful, otherwise None
def fetchTriviaQuestions(
    num_questions: int,
    debug: bool = False,
    category_id: Optional[int] = None
) -> Optional[List[Dict[str, str]]]:
    """Fetch trivia questions from the OpenTrivia Database API"""
    
    if debug:
        print("Entering fetchTriviaQuestions...")
        
    api_url = f"https://opentdb.com/api.php?amount={num_questions}"
    if category_id:
        api_url += f"&category={category_id}"
    
    # query the API
    response = get(api_url)
    
    if response.status_code == 200:
        if debug:
            print("Exiting fetchTriviaQuestions.")
        return response.json()['results']
    # query unsuccessful
    else:
        # output error code
        print(f"Error: {response.status_code}\nFailed to retrieve trivia question(s)")
        if debug:
            print("Exiting fetchTriviaQuestions.")
        return None


def getQuestion(
    category: int,
    question_amount: int,
    debug: bool = False
) -> Tuple[str, str, List[str]]:
    """Fetch a trivia question from the API."""
    
    # basically from trivia.py
    if debug:
        print(f"--Entering getQuestion... Cat {category}, ${question_amount}")
    
    # ensure correct scope
    global API_TOKEN
    
    # init API URL var
    api_url: str = f'https://opentdb.com/api.php?amount=1&token={API_TOKEN}&category={category}&type=multiple'
    
    # check question difficulty, update difficulty
    if question_amount == 100:
        api_url += '&difficulty=easy'
    elif question_amount in [200, 300]:
        api_url += '&difficulty=medium'
    elif question_amount in [400, 500]:
        api_url += '&difficulty=hard'

    if debug:
        print(f"--API URL: {api_url}")
        
    # this is just for if the token needs to be reset, a question will still be returned
    while True:
        # query the API
        response = get(api_url)
        
        if response.status_code == 200:
            data = response.json()
            
            # extract the question and answers from the response
            if data['response_code'] == 0:
                question_data = data['results'][0]
                # separate response into respective vars, also convert any HTML entities to their
                # characters
                question = unescape(question_data['question'])
                correct_answer = unescape(question_data['correct_answer'])
                incorrect_answers = [unescape(ans) for ans in question_data['incorrect_answers']]
                
                if debug:
                    print("--Exiting getQuestion.")
                return question, correct_answer, incorrect_answers
            # reset token if all questions used
            elif data['response_code'] == 4:
                API_TOKEN = resetToken(API_TOKEN,debug)
            else:
                raise ValueError(f"API response error {data['response_code']}")
        else:
            raise ValueError(f"Failed to fetch question from API ({response.status_code})")


def resetToken(
    api_token: str,
    debug: bool = False
) -> str:
    """Reset the session token."""
    
    if debug:
        print("--Entering reset_token...")
        
    response = get(f"https://opentdb.com/api_token.php?command=reset&token={api_token}")
    
    if response.status_code == 200:
        data = response.json()
        if data['response_code'] == 0:
            dumpToken(data['token'],debug)
            if debug:
                print("--Exiting reset_token.")
            return data['token']
        else:
            raise ValueError(f"Error in resetting token: {data['response_message']}")
    else:
        raise ConnectionError(f"HTTP Error: {response.status_code}")


def dumpToken(
    api_token: str,
    debug: bool = False
) -> None:
    """Dumps the API token to an external file."""
    
    if debug:
        print("--Entering dump_token...")
        
    try:
        with open("token",'w') as file:
            file.write(api_token)
            if debug:
                print("Dumped API token to filename 'token'")
    except IOError as e:
        print(f"Error dumping API token (Error: {e})")
        
    if debug:
        print("--Exiting dump_token.")



# 
# 
class Team:
    def __init__(self, name: str) -> None:
        """Represents a trivia team."""
        self.name = name
        self.score = 0

    # pre-condition: points must be initialized to a non-negative integer
    # post-condition: adds the specified points to the team's score
    def updateScore(self, points: int) -> None:
        """Update the team's score by the given points."""
        self.score += points

    def __str__(self) -> str:
        return f"{self.name}: {self.score} points"



# 
# 
class JeopardyGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Jeopardy Game")
        self.debug = False  # Toggle this with a button or menu option for debug mode

        # Main frame for the game board
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack(pady=10)

        # Scoreboard and team management
        self.teams = []
        self.current_team_index = 0
        self.setup_teams()

        # Game setup: categories and questions
        self.MAX_CATEGORIES = 6
        self.categories = self.get_categories()
        self.monetary_values = [100, 200, 300, 400, 500]
        self.table = [[f"${value}" for value in self.monetary_values] for _ in self.categories]

        # Draw the game board
        self.draw_board()

        # Score display
        self.score_frame = tk.Frame(self.root)
        self.score_frame.pack(pady=10)
        self.score_labels = []
        self.update_scores()

    def setup_teams(self):
        # Get team names
        for i in range(2):  # Adjust number of teams as needed
            team_name = simpledialog.askstring("Team Setup", f"Enter Team {i + 1} name:")
            if team_name:
                self.teams.append(Team(team_name))

    def get_categories(self):
        # Sample categories from a predefined list
        category_names = {
            9: "General Knowledge",
            10: "Books",
            11: "Film",
            12: "Music",
            13: "Musicals & Theatres",
            14: "Television",
            15: "Video Games",
            16: "Board Games",
            17: "Science & Nature",
            18: "Computers",
            19: "Mathematics",
            20: "Mythology",
            21: "Sports",
            22: "Geography",
            23: "History",
            24: "Politics",
            25: "Art",
            26: "Celebrities",
            27: "Animals",
            28: "Vehicles",
            29: "Comics",
            30: "Gadgets",
            31: "Japanese Anime & Manga",
            32: "Cartoon & Animation"
        }
        selected_categories = sample(list(category_names.keys()), self.MAX_CATEGORIES)
        return [category_names[num] for num in selected_categories]

    def draw_board(self):
        # Create a grid of buttons for the game board
        for i, category in enumerate(self.categories):
            category_label = tk.Label(self.main_frame, text=category, font=("Arial", 12, "bold"))
            category_label.grid(row=0, column=i, padx=10, pady=5)

            for j, value in enumerate(self.monetary_values):
                button = tk.Button(self.main_frame, text=f"${value}", width=10, height=2,
                                   command=lambda r=i, c=j: self.select_question(r, c))
                button.grid(row=j + 1, column=i, padx=5, pady=5)

    def select_question(self, row, col):
        # Handle question selection
        category = self.categories[row]
        value = self.monetary_values[col]
        self.ask_question(category, value, row, col)

    def ask_question(self, category, value, row, col):
        name_to_number = {v: k for k, v in self.categories.items()}
        #name_to_number = {v: k for k, v in category_names.items()}
        
        # Fetch question data from API (Placeholder function call)
        question, correct_answer, incorrect_answers = getQuestion(name_to_number[category], value)
        all_answers = incorrect_answers + [correct_answer]
        shuffle(all_answers)

        # Question window
        question_window = tk.Toplevel(self.root)
        question_window.title(f"{category} - ${value}")

        question_label = tk.Label(question_window, text=question, wraplength=400, font=("Arial", 14))
        question_label.pack(pady=20)

        # Display possible answers
        for i, answer in enumerate(all_answers):
            answer_button = tk.Button(
                question_window, text=answer, width=40,
                command=lambda a=answer: self.validate_answer(question_window, a, correct_answer, value)
            )
            answer_button.pack(pady=5)

    def validate_answer(self, question_window, answer, correct_answer, value):
        # Validate the answer and update scores
        current_team = self.teams[self.current_team_index]
        if answer == correct_answer:
            messagebox.showinfo("Correct!", "Correct answer!")
            current_team.updateScore(value)
        else:
            messagebox.showerror("Incorrect", "Incorrect answer.")

        question_window.destroy()
        self.update_scores()
        self.switch_team()

    def update_scores(self):
        # Update score labels
        for label in self.score_labels:
            label.destroy()
        self.score_labels = []

        for team in self.teams:
            label = tk.Label(self.score_frame, text=f"{team.name}: {team.score}", font=("Arial", 12))
            label.pack()
            self.score_labels.append(label)

    def switch_team(self):
        # Switch to the next team
        self.current_team_index = (self.current_team_index + 1) % len(self.teams)


"""
def main():
    parser = ArgumentParser(description="Trivia game with optional flags.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("-c", "--correct", action="store_true", help="Display only correct answer.")
    parser.add_argument("-mc", "--multiple-choice", action="store_true", help="Show multiple choice options.")
    parser.add_argument("num_questions", type=int, help="Number of trivia questions to fetch.")
    # process runtime args 
    args = parser.parse_args()
    
    debug: bool = args.debug
    show_correct_answer: bool = args.correct
    show_multiple_choice: bool = args.multiple_choice
    
    if debug:
        print(f"Debug Mode ON\nNumber of questions: {args.num_questions}")
    
    # query available categories
    categories = fetchCategories(debug)
    
    # if we got a successful response from the API
    if categories:
        # list categories
        print("\nAvailable categories:")
        for cat_id, cat_name in categories.items():
            print(f"{cat_id}. {cat_name}")
        
        # user selection
        category_id = int(input("\nSelect a category (0 for any): "))
        # input validation
        while category_id not in categories and category_id != 0:
            category_id = int(input("Invalid category. Please select a valid category number: "))
        
        questions = fetchTriviaQuestions(args.num_questions,debug,category_id)
    # query response unsuccessful, assume no category
    else:
        questions = fetchTriviaQuestions(args.num_questions,debug)

    if questions:
        for i, question_data in enumerate(questions):
            user_answer, correct_answer, all_answers = displayQuestion(question_data,i,debug,show_multiple_choice)
            
            if type(user_answer) == int:
                # mark correct if the input number matches the answer number
                is_correct = all_answers[user_answer-1] == correct_answer
            else:
                # mark correct if the input answer is correct
                is_correct = user_answer == correct_answer

            if is_correct:
                print("Correct!")
            else:
                print("Wrong.", f"The correct answer was: {correct_answer}" if show_correct_answer else "")
"""


if __name__ == "__main__":
    root = tk.Tk()
    app = JeopardyGUI(root)
    root.mainloop()

