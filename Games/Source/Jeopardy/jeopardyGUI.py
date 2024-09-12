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
from random import sample, shuffle        # random category, mix answers
from typing import List                   # variable and function type hinting

API_TOKEN: str = ""
MAX_CATEGORIES: int = 6
# var for each level of question
MONETARY_VALUES = [100,200,300,400,500]
#ANSWER_TIME: int = 30
CATEGORY_NAMES = {
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


# --- LIBRARY -----------------------------------

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
    

"""
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
    except Exception as e:
        messagebox.showerror("Error", f"Error retrieving API token: {e}")
        root.quit()
    
    if debug:
        print(f"--API token: {API_TOKEN}")
    
    # get team names from user
    teams = setupTeams()

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

    # --- TABLE CREATION ------------------------

    # init with monetary values for each category
    table = [[f"${value}" for value in MONETARY_VALUES] for _ in selected_category_names]

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
"""




from kivy.app import App
from kivy.uix.label import Label
from kivy.uix.gridlayout import GridLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
#from kivy.uix.widget import Widget
#from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.uix.boxlayout import BoxLayout
#from kivy.core.window import Window

from jeopardy_api import JeopardyAPI

class MyLayout(GridLayout):
    #MAX_CATEGORIES = ObjectProperty(MAX_CATEGORIES)
    #question_amount = ObjectProperty(None)
    #category_name = ObjectProperty(None)
    #category_index = ObjectProperty(None)
    
    # init infinite keywords
    # pre-condition: 
    # post-condition: 
    def __init__(self, **kwargs):
        # call grid layout constructor
        super(MyLayout, self).__init__(**kwargs)
        
        ############ INIT CLASS VARS ############
        # todo: incorporate debug flags and optional filename
        self.steal_attempted: bool = False
        # we also have the following:
        # api_token (str)
        # team1, team2, current_team, question_amount, category_name (str)
        # name_to_number (List[int])
        # selected_category_names (List[str])
        # category_index (int)
        # question, correct_answer, incorrect_answers
        # 
        # stuff pertaining to the GUI
        # cols (int), buttons (List), width (int)
        
        # init teams popup
        # we need to first create the widgets, then send it to a popup
        # this is the 'top'/outer layout, it should have 1 columns
        team_init_popup = GridLayout(cols=1,spacing=10,padding=10)
        
        # team 1 widgets
        team_input = GridLayout(cols=2,spacing=10,padding=[10,10])
        team_input.add_widget(Label(text="Team 1 name:", size_hint_y=None, height=30, valign='middle'))
        team_1_input = TextInput(multiline=False, size_hint_y=None, height=30)
        team_input.add_widget(team_1_input)
        # team 2 widgets
        team_input.add_widget(Label(text="Team 2 name:", size_hint_y=None, height=30, valign='middle'))
        team_2_input = TextInput(multiline=False, size_hint_y=None, height=30)
        team_input.add_widget(team_2_input)
        # add to main widget
        team_init_popup.add_widget(team_input)
        
        # this is all button stuff
        button_container = GridLayout(cols=1,size_hint_y=None, height=50, spacing=10, padding=[100,10])
        submit_button = Button(
            text="Done",
            background_color=(0,1,0,1),
            size_hint_x=None,
            width=100,
            pos_hint={'center': 0.5}
        )
        # todo: button is too close to input fields
        submit_button.bind(on_press=lambda instance: self.saveTeamNames(team_1_input.text, team_2_input.text, popup))
        button_container.add_widget(submit_button)
        
        # add button stuff to popup
        team_init_popup.add_widget(button_container)
        
        # add button to popup
        popup = Popup(
            title='Enter team names',
            title_size='20sp',
            auto_dismiss=False,
            content=team_init_popup,
            size_hint=(0.45, 0.35),
            #separator_height=2
            pos_hint={'center_x': 0.5, 'center_y': 0.5}
        )
        # call popup
        popup.open()
    
    # pre-condition: 
    # post-condition: 
    def saveTeamNames(self, team1, team2, popup):
        # Save or use team names here
        self.team1 = team1
        self.team2 = team2
        self.current_team = team1
        #if debug:
        print(f"Team 1: {team1}, Team 2: {team2}")
        popup.dismiss()

        self.api_token: str = JeopardyAPI()
        self.mainWindow()
    
    # pre-condition: 
    # post-condition: 
    def mainWindow(self):
        # set cols
        self.cols = MAX_CATEGORIES
        
        # --- CATEGORIES AND VARIABLES --------------
        # pick 6 random numbers from the range 9-32
        selected_category_indices = sample(list(range(9,33)), MAX_CATEGORIES)
        # create a reverse mapping for API call
        self.name_to_number = {v: k for k, v in CATEGORY_NAMES.items()}
        # load selected categories into list
        self.selected_category_names = [CATEGORY_NAMES[num] for num in selected_category_indices]
        # init with monetary values for each category
        #if debug:
        print(f"Categories: {self.selected_category_names} (indices {selected_category_indices})")
        
        # add widgets
        for category in self.selected_category_names:
            #self.add_widget(Label(text=category))
            category_label = Label(
                text=category,
                font_size=18,
                size_hint_y=None,
                height=80,
                text_size=(120, None), # Set width constraint for wrapping
                halign='center',
                valign='middle'
            )
            self.add_widget(category_label)
        
        # add buttons for each value in each category
        self.buttons = []
        for value in MONETARY_VALUES:
            for category in self.selected_category_names:
                # create button
                button = Button(
                    text=f"${value}",
                    size_hint=(None, None),
                    height=60, # 50
                    width=120, # 75
                    background_color=(0, 0, 1, 1),  # Blue background similar to Jeopardy
                    color=(1, 1, 0, 1),  # Yellow text color
                    bold=True
                )
                # bind button
                button.bind(on_press=lambda instance, category=category: self.press(instance,category))
                self.add_widget(button)
                self.buttons.append((button, category))
    
    # pre-condition: 
    # post-condition: 
    def press(self, instance, category):
        self.question_amount = instance.text
        self.category_name = category
        self.category_index = self.name_to_number[category]
        #if debug:
        print(f"Pressed {self.question_amount} in category {self.category_name} (index {self.category_index})")
        
        # query api for data
        try:
            self.question, self.correct_answer, self.incorrect_answers = self.api_token.getQuestion(
                category=self.category_index,
                question_amount=self.question_amount
            )
        except ValueError as ve:
            print(f"Error fetching question: {ve}")
            return
        except ConnectionError as ce:
            print(f"Network error: {ce}")
            return
        # Check that we got valid info
        if not self.question or not self.correct_answer or not self.incorrect_answers:
            print("Error: Missing question or answer data.")
            return
        
        # display
        self.displayQuestion()
    
    # pre-condition: 
    # post-condition: 
    def displayQuestion(self, is_steal=False):
        # 
        if not is_steal:
            self.steal_attempted = False
        if is_steal:
            print(f"Steal opportunity for {self.current_team}!")
        
        all_answers = self.incorrect_answers + [self.correct_answer]
        
        box = BoxLayout(orientation='vertical',spacing=10,padding=10)
        # todo: ensure line wrapping does not exceed popup box size
        question_label = Label(
            text=self.question,
            font_size=30,
            size_hint_y=None,
            height=150,
            text_size=(self.width*0.8,None),
            halign='center',
            valign='middle'
        )
        box.add_widget(question_label)
        
        # Create buttons for each answer
        # todo: line wrapping and ensure it does not exceed button size
        for answer in all_answers:
            answer_button = Button(text=answer,font_size=25, size_hint_y=None, height=50)
            answer_button.bind(on_press=lambda instance, ans=answer: self.checkAnswer(ans, self.correct_answer, popup))
            box.add_widget(answer_button)

        # Create the popup
        # todo: make title bigger
        popup_title = f"{self.current_team}, select your answer"
        popup = Popup(
            title=popup_title,
            content=box,
            auto_dismiss=False,
            size_hint=(0.8, 0.8)
        )
        popup.open()
    
    # pre-condition: 
    # post-condition: 
    def checkAnswer(self, selected_answer, correct_answer, popup):
        # Check if the selected answer is correct
        if selected_answer == correct_answer:
            result_text = "Correct!"
            
            # Display result in a new popup
            result_popup = Popup(
                title='Result',
                content=Label(text=result_text),
                size_hint=(0.5, 0.5))
            result_popup.open()

            # Close the question popup
            popup.dismiss() 
        else:
            # Handle incorrect answer and check for steal opportunity
            popup.dismiss()
            if not self.steal_attempted:
                # Allow a single steal attempt
                self.steal_attempted = True
                # Switch to the other team for stealing
                self.current_team = self.team2 if self.current_team == self.team1 else self.team1
                self.displayQuestion(is_steal=True)
            else:
                # End question if the steal has already been attempted
                result_popup = Popup(
                    title='Result',
                    content=Label(text="Incorrect."),
                    size_hint=(0.5, 0.5)
                )
                result_popup.open()
                popup.dismiss()  # Close the question popup


class JeopardyApp(App):
    def build(self):
        return MyLayout()


if __name__ == '__main__':
    JeopardyApp().run()


# todo:
# - change teams after correct
# - remove buttons for used questions
# - display points
# - api token filename var
# - add CLI arg flags
# - maybe add something in api_token / jeopardyAPI init that checks for internet 
# 
# sounds? 
# timer?