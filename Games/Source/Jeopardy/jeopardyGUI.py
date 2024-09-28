# William Wadsworth
# Created: .2024
# Initial release: .2024
# 
# This script is Jeopardy with Trivia questions via the Trivia API from Open Trivia Database
# (documentation: https://opentdb.com/api_config.php). Notes about use of the API will be found in
# the documentation, and a simple demonstration can be found in trivia.py. Debug output is labeled
# with '[DEBUG]'. An internet connection is required for the program to work. Note that if a token
# file is present, the program will overwrite the old token with the new one.
# 
# Usage: python3 jeopardyGUI.py

from sys import stderr#, exit
#from argparse import ArgumentParser       # parse runtime args without spaghetti logic
from random import sample, shuffle        # random category, mix answers
#from typing import Optional, List         # variable and function type hinting

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



# --- LIBRARY -----------------------------------
"""
# pre-condition: none
# post-condition: each team name and score is output
def displayScores(
    teams: List[Team],
    debug: bool = False
) -> None:
    ""Display the current scores of all teams.""
    
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
    ""Check if all questions have been answered.""
    
    return all(item == " " for row in table for item in row)


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
    ""Validate the team's answer and update the score if correct.""
    
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
from kivy.core.window import Window
from kivy.clock import Clock
from kivy.uix.checkbox import CheckBox

from os import path
from jeopardy_api import JeopardyAPI
from team import Team

class MyLayout(GridLayout):
    #MAX_CATEGORIES = ObjectProperty(MAX_CATEGORIES)
    #question_amount = ObjectProperty(None)
    #category_name = ObjectProperty(None)
    #category_index = ObjectProperty(None)
    
    # init infinite keywords
    # pre-condition: 
    # post-condition: 
    def __init__(self, **kwargs) -> None:
        # call grid layout constructor
        super(MyLayout, self).__init__(**kwargs)
        
        ############ INIT CLASS VARS ############
        # todo: incorporate optional filename
        self.API_TOKEN: str = ""
        self.DEBUG: bool = False
        
        self.steal_attempted: bool = False
        self.previous_winner: str = ""
        self.token_filename: str = ""
        # we also have the following:
        # team1, team2, current_team (Team),
        # current_category (str), current_question_points (int)
        # name_to_number (List[int])
        # selected_category_names (List[str])
        # category_index (int)
        # question, correct_answer (str), incorrect_answers (List[str])
        # 
        # stuff pertaining to the GUI
        # cols (int), buttons (List), width (int)
        # team1_score_label, team2_score_label (Label)
        # team1_turn_label, team2_turn_label
        
        self.initPopup()
    
    # pre-condition: 
    # post-condition: 
    def initPopup(self) -> None:
        if self.DEBUG:
            print("[DEBUG] Entering initPopup...")
        
        # init teams popup
        # we need to first create the widgets, then send it to a popup
        # this is the 'top'/outer layout, it should have 1 columns
        team_init_popup = GridLayout(cols=1,spacing=10,padding=10)
        
        team_init_popup.add_widget(Label(
            text="Note: Team 1 goes first",color=(1,0,0,1),bold=True,
            italic=True,underline=True,size_hint_y=None, height=10, valign='middle'))
        
        # team 1 widgets
        popup_content = GridLayout(cols=3,spacing=10,padding=[10,10])
        popup_content.add_widget(Label(text="Team 1:", size_hint_y=None, height=30, valign='middle'))
        team_1_input = TextInput(multiline=False, size_hint_y=None, height=30)
        popup_content.add_widget(team_1_input)
        # debug checkbox label
        popup_content.add_widget(Label(text="Debug",size_hint_y=None,height=30,valign="middle"))
        # team 2 widgets
        popup_content.add_widget(Label(text="Team 2:", size_hint_y=None, height=30, valign='middle'))
        team_2_input = TextInput(multiline=False, size_hint_y=None, height=30)
        popup_content.add_widget(team_2_input)
        # debug checkbox
        popup_content.add_widget(CheckBox(size_hint_y=None,height=30,allow_no_selection=True,
            ))
        # active, 
        # optional filename
        popup_content.add_widget(Label(text="Filename:",size_hint_y=None,height=30,valign="middle"))
        filename_input = TextInput(multiline=False,size_hint_y=None,height=30)
        popup_content.add_widget(filename_input)
        
        # add to main widget
        team_init_popup.add_widget(popup_content)
        
        # this is all button stuff
        button_container = GridLayout(cols=1,size_hint_y=None, height=30,spacing=10)
        submit_button = Button(text="Done",background_color=(0,1,0,1),size_hint_x=None)
        submit_button.bind(on_press=lambda instance: self.saveTeamNames(team_1_input.text,team_2_input.text,filename_input,popup))
        button_container.add_widget(submit_button)
        # add button stuff to popup
        for i in range(4): # 4 because we have an empty spot in the row above
            if i == 2: popup_content.add_widget(button_container) # V all this stuff to keep alignment
            else: popup_content.add_widget(Label(text="",size_hint_y=None,height=30,valign="middle"))
        
        # add button to popup
        popup = Popup(
            title='Enter team names',
            title_size='20sp',
            auto_dismiss=False,
            content=team_init_popup,
            size_hint=(0.50, 0.48), # x,y
            pos_hint={'center_x': 0.5, 'center_y': 0.5})
        popup.open()
        
        if self.DEBUG:
            print("[DEBUG] Exiting initPopup.")
    
    # pre-condition: 
    # post-condition: 
    def checkboxClick(self, checkbox: CheckBox):
        if checkbox.active:
            self.DEBUG = True
    
    # pre-condition: 
    # post-condition: 
    def saveTeamNames(self, team1: str, team2: str, filename: str, popup: Popup) -> None:
        if self.DEBUG:
            print("[DEBUG] Entering saveTeamNames...")
            
        #token_filename: str = ""
        # assume that whatever is here is the possible filename
        self.token_filename = filename if filename and path.exists(filename) else ""
        
        # secret debug and token filename input toggle 
        #if team1.lower() == "debug":
        #    print("Debug mode: ON")
        #    self.DEBUG = True
        #    popup.dismiss()
        #    self.initPopup()
        #    return

        # Normal team name entry
        #if not team1.strip() or not team2.strip():
        #    print("Error: Both team names must be provided.")
        #    return  # Return without dismissing popup if names are missing
        
        self.team1 = Team(team1)
        self.team2 = Team(team2)
        self.current_team = self.team1
        
        if self.DEBUG:
            print(f"[DEBUG] Team 1: {team1}, Team 2: {team2}")
        popup.dismiss()
        
        #print("test1")
        if self.token_filename:
            #print("test2")
            self.API_TOKEN = JeopardyAPI(self.token_filename,self.DEBUG)
        else:
            self.API_TOKEN = JeopardyAPI(self.DEBUG)
        if self.DEBUG:
            print("[DEBUG] Exiting saveTeamNames.")
        self.mainWindow()
    
    # pre-condition: 
    # post-condition: 
    def mainWindow(self) -> None:
        if self.DEBUG:
            print("[DEBUG] Entering mainWindow...")
        
        # set window size
        Window.size = (780,500)
        
        # set cols
        self.cols = MAX_CATEGORIES
        self.team1_turn_label = Label(text="^")
        self.team2_turn_label = Label(text="")
        self.team1_score_label = Label(text=f"{self.team1.score} pts")
        self.team2_score_label = Label(text=f"{self.team2.score} pts")
        # labels defined here so we can update the text later
        
        # ROWS
        # row 1 : team names and points (cols 2,5)
        # row 2 : current team symbol
        # row 3 : categories
        # row 4-8 : buttons
        
        # first row: team names and points
        for i in range(MAX_CATEGORIES):
            if i == 1: self.add_widget(Label(text=f"Team 1: {self.team1.name}"))
            elif i == 2: self.add_widget(self.team1_score_label)
            elif i == 3: self.add_widget(Label(text=f"Team 2: {self.team2.name}"))
            elif i == 4: self.add_widget(self.team2_score_label)
            else: self.add_widget(Label(text="")) # be empty
        # second row: space with current team denoted
        # init to team 1 on first run
        for i in range(MAX_CATEGORIES):
            if i == 1: self.add_widget(self.team1_turn_label)
            elif i == 3: self.add_widget(self.team2_turn_label)
            else: self.add_widget(Label(text=""))
        
        # --- CATEGORIES AND VARIABLES --------------
        # todo : seed rng?
        # pick 6 random numbers from the range 9-32
        selected_category_indices = sample(list(range(9,33)), MAX_CATEGORIES)
        # create a reverse mapping for API call
        self.name_to_number = {v: k for k, v in CATEGORY_NAMES.items()}
        # load selected categories into list
        self.selected_category_names = [CATEGORY_NAMES[num] for num in selected_category_indices]
        # init with monetary values for each category
        if self.DEBUG:
            print(f"[DEBUG] Categories: {self.selected_category_names} (indices {selected_category_indices})")
        
        # add widgets
        for category in self.selected_category_names:
            category_label = Label(
                text=category,
                font_size=18,
                size_hint_y=None,
                height=80,
                text_size=(150, None),
                halign='center',
                valign='middle')
            self.add_widget(category_label)
        
        # add buttons for each value in each category
        self.buttons = []
        for value in MONETARY_VALUES:
            for category in self.selected_category_names:
                # create button
                button = Button(
                    text=f"${value}",
                    size_hint=(None, None),
                    height=64,
                    width=130,
                    background_color=(0, 0, 1, 1),
                    color=(1, 1, 0, 1),
                    bold=True)
                # bind button
                # NOTE: button goes mainWindow -> press -> displayQuestion -> checkAnswer for removal
                button.bind(on_press=lambda instance, btn=button, category=category: self.press(btn,category))
                self.add_widget(button)
                self.buttons.append((button, category))
        
        if self.DEBUG:
            print("[DEBUG] Exiting mainWindow.")
    
    # pre-condition: 
    # post-condition: 
    def press(self, question_button: Button, category: str) -> None:
        if self.DEBUG:
            print("[DEBUG] Entering press...")
            
        self.current_question_points = int(question_button.text.strip('$')) # monetary value (includes '$')
        self.current_category = category                    # category name
        self.category_index = self.name_to_number[category] # category index
        if self.DEBUG:
            print(f"[DEBUG] Pressed {self.current_question_points} in category {category} (index {self.category_index})")
        
        # query api for data
        # todo : move to function?
        try:
            self.question, self.correct_answer, self.incorrect_answers = self.API_TOKEN.getQuestion(
                category=self.category_index,
                question_amount=self.current_question_points)
        except ValueError as ve:
            stderr.write(f"Error fetching question: {ve}")
            return
        except ConnectionError as ce:
            stderr.write(f"Network error: {ce}")
            return
        # Check that we got valid info
        if not self.question:
            stderr.write("Error: Missing question data.")
            return
        elif not self.correct_answer or not self.incorrect_answers:
            stderr.write("Error: Missing answer data.")
            return
        
        if self.DEBUG:
            print(f"Next turn: {self.current_team}\n[DEBUG] Exiting press.")
        self.displayQuestion(question_button)
    
    # pre-condition: 
    # post-condition: 
    def displayQuestion(self, question_button: Button, is_steal: bool = False) -> None:
        if self.DEBUG:
            print(f"[DEBUG] Entering displayQuestion...\nCurrent turn: {self.current_team}")
        
        all_answers = self.incorrect_answers + [self.correct_answer]
        
        #if not is_steal: # it is not a steal opportunity
        #    self.steal_attempted = False
        if is_steal: # it is a steal opportunity
            print(f"Steal opportunity for {self.current_team.name}!")
        else: # it is not a steal opportunity
            self.steal_attempted = False
            shuffle(all_answers)
        
        box = BoxLayout(orientation='vertical',spacing=10,padding=[10,20,10,10])
        # todo: ensure line wrapping does not exceed popup box size
        question_label = Label(
            text=self.question,
            font_size=18,
            size_hint_y=None,
            height=100,
            text_size=(Window.width*0.75,None), # text should wrap at 75% window width
            halign='center',
            valign='middle')
        box.add_widget(question_label)
        
        # Create buttons for each answer
        # todo: line wrapping and ensure it does not exceed button size
        for answer in all_answers:
            answer_button = Button(text=answer,font_size=18, size_hint_y=None, height=40)
            answer_button.bind(on_press=lambda instance, ans=answer: self.checkAnswer(ans,popup,question_button))
            box.add_widget(answer_button)

        # Create the popup
        popup = Popup(
            title=f"{self.current_team}, select your answer",
            content=box,
            auto_dismiss=False,
            size_hint=(0.9, 0.7),
            title_size='15sp')
        popup.open()
        # todo: popup size is off
        
        if self.DEBUG:
            print("[DEBUG] Exiting displayQuestion.")
    
    # pre-condition: 
    # post-condition: 
    def checkAnswer(self, selected_answer: str, popup: Popup, question_button: Button) -> None:
        if self.DEBUG:
            print("[DEBUG] Entering checkAnswer...")
        
        if selected_answer == self.correct_answer:
            # update score
            self.current_team.updateScore(int(question_button.text.strip('$')),self.DEBUG)
            # update scores in main window
            self.team1_score_label.text = f"{self.team1.score} pts"
            self.team2_score_label.text = f"{self.team2.score} pts"
            # Schedule UI refresh on the next frame
            Clock.schedule_once(lambda dt: self.team1_score_label.canvas.ask_update())
            Clock.schedule_once(lambda dt: self.team2_score_label.canvas.ask_update())
            
            # swap teams for next turn (if it is not a steal)
            if not self.steal_attempted:
                self.switchTurn()
            # todo: add logic that prevents turn switching if steal is correct
            
            # Display result in a new popup
            result_popup = Popup(
                title='Result',
                content=Label(text="Correct!"),
                size_hint=(0.5, 0.5))
            result_popup.open()
            
            popup.dismiss()
            
            # disable button
            question_button.disabled = True
            question_button.background_color = (0.5,0.5,0.5,1) # change color to indicate not valid
            # or hide button
            #question_button.opacity = 0
            #question_button.disabled = True
        else:
            popup.dismiss()
            if not self.steal_attempted: # it is a steal attempt
                self.steal_attempted = True
                # Switch to the other team for stealing
                self.switchTurn()
                self.displayQuestion(question_button,is_steal=True)
            else:
                # End question if the steal has already been attempted
                result_popup = Popup(
                    title='Result',
                    content=Label(text="Incorrect."),
                    size_hint=(0.5, 0.5))
                result_popup.open()
                
                popup.dismiss()
                # clear the button
                question_button.disabled = True
                question_button.background_color = (0.5,0.5,0.5,1)
        
        if self.DEBUG:
            print("[DEBUG] Exiting checkAnswer.")
        
    # pre-condition: 
    # post-condition: 
    def switchTurn(self) -> None:
        if self.DEBUG:
            print("[DEBUG] Entering switchTurn...")
            
        # Switch to the other team after the turn ends
        self.current_team = self.team2 if self.current_team == self.team1 else self.team1
        #if self.DEBUG:
        #    print(f"Next team: {self.current_team}")

        # Update the '^' symbol to indicate the current team
        self.team1_turn_label.text, self.team2_turn_label.text = self.team2_turn_label.text, self.team1_turn_label.text
        Clock.schedule_once(lambda dt: self.team1_turn_label.canvas.ask_update())
        Clock.schedule_once(lambda dt: self.team2_turn_label.canvas.ask_update())

        if self.DEBUG:
            print("[DEBUG] Exiting switchTurn.")

class JeopardyApp(App):
    def build(self):
        return MyLayout()


if __name__ == "__main__":
    JeopardyApp().run()


# todo:
# problem with switchTurn where it retains a team's turn if their turn was stolen
# - CLI arg flags
# - end game popup (play again, symbol denotes previous winner)
# - something in jeopardy_api init that checks for internet 
# 
# sounds? 
# timer?