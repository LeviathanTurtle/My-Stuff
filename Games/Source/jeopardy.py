# William Wadsworth
# Created: 7.22.2024
# Initial release: 7.28.2024
# Updated 8.16.2024: PEP 8 Compliance
# Updated 8.24.2024: added token persistence via external file
# Updated 8.29.2024: now uses Argparse instead of spaghetti logic for runtime arguments, moved team
#                    answering logic to external function, reduced global variables
# 
# This script is Jeopardy with Trivia questions via the Trivia API from Open Trivia Database
# (documentation: https://opentdb.com/api_config.php). Notes about use of the API will be found in
# the documentation, and a simple demonstration can be found in trivia.py. Debug output is labeled
# with '--'. An internet connection is required for the program to work. Note that if a token file
# is present, the program will overwrite the old token with the new one.
# 
# Usage: python3 jeopardy.py [-token <filename>] [-d] [-c] [-mc]
# Run 'python3 jeopardy.py' with "-h" or "--help" for argument descriptions
# 
# Potential planned features:
# - timer
# - sound effects?
# - tab autocomplete?

from sys import stderr, exit
from argparse import ArgumentParser       # parse runtime args without spaghetti logic
from requests import get                  # make a HTTP GET request
from html import unescape                 # convert anomalous characters to unicode
from random import sample, shuffle        # random category, mix answers
from typing import Tuple, List, Optional  # variable and function type hinting

API_TOKEN: str = ""
MAX_CATEGORIES: int = 6
#ANSWER_TIME: int = 30

# --- API TOKEN HANDLING ------------------------

# pre-condition: an internet connection
# post-condition: returns the token generated by the API
def getSessionToken(
    debug: bool = False,
    token_filename: Optional[str] = None
) -> str:
    """Generate a new session token for the trivia API."""
    
    if debug:
        print("--Entering get_session_token...")
        
    file_token: Optional[str] = None
        
    if token_filename is not None:
        try:
            # file stuff
            with open(token_filename, 'r') as file:
                file_token = file.read().strip()
        except FileNotFoundError:
            print(f"Warning: file '{token_filename}' not found. Generating a new token")
    
    # return the token from the file if it was read
    if file_token:
        # dump to file
        dumpToken(file_token,debug)
        if debug:
            print("--Exiting get_session_token.")
        return file_token
    
    # otherwise, assume we still need a token, so generate a new one
    response = get("https://opentdb.com/api_token.php?command=request")
    
    if response.status_code == 200:
        data = response.json()
        if data['response_code'] == 0:
            dumpToken(data['token'],debug)
            if debug:
                print("--Exiting get_session_token.")
            return data['token']
        else:
            raise ValueError(f"Error in resetting token: {data['response_message']}")
    else:
        raise ConnectionError(f"HTTP Error: {response.status_code}")


# pre-condition: an internet connection, api_token must be initialized as a string representing the
#                API token
# post-condition: returns the re-generated token by the API
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


# pre-condition: api_token must be initialized as a string representing the API token
# post-condition: outputs the API token to a file named 'token'
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


# --- LIBRARY -----------------------------------

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


# pre-condition: none
# post-condition: the current jeopardy board is printed
def displayTable(
    selected_category_names: List[str],
    table: List[List[str]],
    debug: bool = False
) -> None:
    """Display the current state of the trivia table."""
    
    if debug:
        print("--Entering display_table...")
    
    # calculate the width for each column (including padding)
    minimum_space: int = 8
    column_widths = [max(len(name), minimum_space)+4 for name in selected_category_names]
    
    # header
    header = "|".join([f" {name:^{column_widths[i]-2}} " for i, name in enumerate(selected_category_names)])
    print(f"|{header}|")
    
    # separator
    print("-" * (len(header) + len(selected_category_names) - 4))
    
    # display each row of the table
    for row in zip(*table):
        row_display = "|".join([f" {item:^{column_widths[i]-2}} " for i, item in enumerate(row)])
        print(f"|{row_display}|")
    
    if debug:
        print("--Exiting display_table.")


# pre-condition: category must be initialized as a string representing a valid category, value must
#                be initialized to a positive, non-negative integer representing a valid question
#                value
# post-condition: the question index in the board is replaced with whitespace
def updateTable(
    table: List[List[str]],
    category: str,
    value: int,
    selected_category_names: List[str],
    monetary_values: List[int],
    debug: bool = False
) -> None:
    """Update the trivia table by marking a question as answered."""
    
    if debug:
        print("--Entering update_table...")
    
    col_index = selected_category_names.index(category)
    row_index = monetary_values.index(value)
    table[col_index][row_index] = " "
    
    if debug:
        print("--Exiting update_table.")


# pre-condition: category must be initialized as a string, question_amount must be initialized to a
#                number [9,32], API_TOKEN must be initialized to a valid API token
# post-condition: a tuple containing the question (string), the correct answer (string), and a list
#                 of three incorrect answers (list of strings)
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


# pre-condition: none
# post-condition: each team name and score is output
def displayScores(
    teams: List[Team],
    debug: bool = False
) -> None:
    """Display the current scores of all teams."""
    
    if debug:
        print("--Entering display_scores...")
    
    for team in teams:
        print(team)
    
    if debug:
        print("--Exiting display_scores.")


# pre-condition: none
# post-condition: True is returned if every question on the board is whitespace, otherwise False is
#                 returned
def allQuestionsAnswered(
    table: List[List[str]],
) -> bool:
    """Check if all questions have been answered."""
    
    return all(item == " " for row in table for item in row)


# pre-condition: current_index must be initialized to a positive, non-negative integer
# post-condition: returns the index of the next team
def switchTeam(
    current_index: int,
    total_teams: int
) -> int:
    """Switch to the next team."""
    
    return (current_index + 1) % total_teams


# pre-condition: 
# post-condition: 
def validateAnswer(
    team: Team,
    question_amount: int,
    correct_answer,
    all_answers,
    show_multiple_choice: bool,
    debug: bool = False
) -> bool: 
    """Validate the team's answer and update the score if correct."""
    
    if debug:
        print("--Entering validateAnswer...")
        
    # repeat this until we get correct input
    while True:
        # get team's answer input
        team_answer = input("\n: ")#.strip()
        
        # if we show multiple choice options, the user inputs a number
        if show_multiple_choice:
            if team_answer.isdigit() and int(team_answer) in range(1, len(all_answers)+1):
                if all_answers[int(team_answer)-1] == correct_answer:
                    print("Correct!")
                    team.updateScore(question_amount)
                    if debug:
                        print("--Exiting validateAnswer.")
                    return True
                else:
                    print("Incorrect.")
                    if debug:
                        print("--Exiting validateAnswer.")
                    return False
            else:
                print("Invalid choice. Please enter a valid number.")
        else:
            # no numbers, input string
            if team_answer.lower() == correct_answer.lower():  # case sensitivity
                print("Correct!")
                team.updateScore(question_amount)
                if debug:
                    print("--Exiting validateAnswer.")
                return True
            else:
                print("Incorrect.")
                if debug:
                    print("--Exiting validateAnswer.")
                return False
    

def main():
    # python3 jeopardy.py [-token <filename>] [-d] [-c] [-mc]
    parser = ArgumentParser(description="Jeopardy game with optional flags.")
    parser.add_argument("-token","--token",metavar="FILENAME",help="Filename containing the API token.")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("-c", "--correct", action="store_true", help="Display only correct answer.")
    parser.add_argument("-mc", "--multiple-choice", action="store_true", help="Show multiple choice options.")
    # process runtime args 
    args = parser.parse_args()
    
    debug: bool = args.debug
    show_correct_answer: bool = args.correct
    show_multiple_choice: bool = args.multiple_choice
    
    if debug:
        print(f"Debug mode: ON\n--Show answer: {show_correct_answer}, Show MC: {show_multiple_choice}")
    
    try:
        API_TOKEN = getSessionToken(debug,args.token) if args.token else getSessionToken(debug)
    except FileNotFoundError:
        stderr.write(f"Error: Specified token file '{args.token}' not found.")
        exit(1)
    except Exception as e:
        stderr.write(f"Error retrieving API token: {e}")
        exit(1)
    
    if debug:
        print(f"--API token: {API_TOKEN}")
    
    # get team names from user
    team_names = [input(f"Enter Team {i+1}'s name: ") for i in range(2)]
    # convert to Team object
    teams = [Team(name) for name in team_names]

    # --- CATEGORIES AND VARIABLES --------------

    # pick 6 random numbers from the range 9-32
    selected_categories = sample(list(range(9,33)), MAX_CATEGORIES)
    # list of categories and their numerical pairing for the API call
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

    # create a reverse mapping for API call
    name_to_number = {v: k for k, v in category_names.items()}

    # load selected categories into list
    selected_category_names = [category_names[num] for num in selected_categories]

    # var for each level of question
    monetary_values = [100,200,300,400,500]

    # --- TABLE CREATION ------------------------

    # init with monetary values for each category
    table = [[f"${value}" for value in monetary_values] for _ in selected_category_names]

    # --- MAIN LOOP -------------------------------------------------------------------------------

    # get the first team
    #current_team: str = input("Enter the team to go first: ")
    current_team_index = team_names.index(input("Enter the team to go first: "))

    displayTable(selected_category_names, table, debug)

    while not allQuestionsAnswered(table):
        current_team = teams[current_team_index]
        print(f"\nIt's {current_team.name}'s turn.")

        # --- CATEGORY SELECTION ----------------
        # select category and question 
        try:
            selection_category, selection_question = input("Select a category and question: ").rsplit(' ', 1)
            selection_question = int(selection_question)
            # check int selection
            while selection_question not in monetary_values:
                selection_question = int(input("Please enter 100, 200, 300, 400, or 500: "))
        except ValueError:
            print("Invalid, please enter the category and question in the format 'category question'.")
            continue
        # input validation
        if selection_category not in selected_category_names or selection_question not in monetary_values:
            print("Invalid category or question value. Please try again.")
            continue
        
        # check if the question is already answered, input validation
        col_index = selected_category_names.index(selection_category)
        row_index = monetary_values.index(selection_question)
        if table[col_index][row_index] == " ":
            print("This question has already been answered. Please select another.")
            continue
        
        # --- API CALL --------------------------
        # fetch question and answers from API
        try:
            category_number = name_to_number[selection_category]
            question, correct_answer, incorrect_answers = getQuestion(category_number,selection_question,debug)
        except ValueError as e:
            print(f"Value error encountered: {e}. Please select another question.")
            continue
        except KeyError:
            print(f"Category '{selection_category}' not found. Please select another category.")
            continue
        except Exception as e:
            print(f"An error occurred while fetching the question: {e}.")
            continue
        
        # group all answers, mix
        all_answers = incorrect_answers + [correct_answer]
        shuffle(all_answers)
        
        # display the question and answers
        print(f"\n  --- {selection_category} - ${selection_question} ---\n{question}")
        if show_multiple_choice:
            for i, answer in enumerate(all_answers, 1):
                print(f"{i}. {answer}")
        
        # --- ANSWERING -------------------------
        # get the team's answer, validate and update the score
        if not validateAnswer(current_team,selection_question,correct_answer,all_answers,show_multiple_choice,debug):
            # switch to other team for a chance to steal
            other_team_index = switchTeam(current_team_index, len(teams))
            other_team = teams[other_team_index]
            print(f"It's {other_team.name}'s chance to steal.")

            # get other team's answer
            if not validateAnswer(other_team,selection_question,correct_answer,all_answers,show_multiple_choice,debug) and \
               show_correct_answer:
                print(f"The answer is {correct_answer}")
        
        # update the table to mark the question as answered
        updateTable(table,selection_category,selection_question,selected_category_names,monetary_values,debug)
        
        # switch team index to next team
        current_team_index = (current_team_index+1) % len(teams)
        
        # show updated scores
        displayScores(teams,debug)
        print()
        displayTable(selected_category_names, table, debug)

    print("\n\nFINAL SCORES:")
    displayScores(teams,debug)


if __name__ == "__main__":
    main()


# Some fun headaches (bugs) that I came across
# 
# 1. I added validateAnswer after demoting debug from a global var but before demoting
#    show_multiple_choice and show_correct_answer. Because I still wanted to use
#    show_multiple_choice in this function, I added it as a parameter in the function definition,
#    but did not update the calls in the main program. As a reminder, the function takes six
#    arguments: team, question_amount, correct_answer, all_answers, show_multiple_choice, and
#    debug, but the bugged call only had five (all but show_multiple_choice). As a result, the
#    debug boolean took the place of show_multiple_choice, meaning it was set to True when it
#    should not have, and the default debug parameter being False also meant the debug statements
#    were not hit. Ultimately, it meant the program was asking the user for an integer, when they
#    did not specify they wanted multiple choice. 
# 
# 2. 