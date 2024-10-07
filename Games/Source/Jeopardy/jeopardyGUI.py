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

from sys import stderr, exit
#from argparse import ArgumentParser       # parse runtime args without spaghetti logic
from random import sample, shuffle        # random category, mix answers
#from typing import Optional, List         # variable and function type hinting
from traceback import format_exc

MAX_CATEGORIES: int = 1
# var for each level of question
#MONETARY_VALUES = [100,200,300,400,500]
MONETARY_VALUES = [100]
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
from kivy.clock import Clock as kivy_clock
from kivy.uix.checkbox import CheckBox

from kivy.core.audio import SoundLoader

from os.path import exists, getctime
from jeopardy_api import JeopardyAPI
from team import Team
from debug_logger import DebugLogger
from glob import glob
from re import match
from time import time
from typing import Optional

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
        Window.size = (780,500)
        
        ############ INIT CLASS VARS ############
        self.API_TOKEN: str = ""
        #self.DEBUG: bool = False
        self.logger = DebugLogger()
        
        self.is_steal: bool = False
        self.PREVIOUS_WINNER: str = self.checkPreviousWinner()
        self.token_filename: str = ""
        self.use_audio: bool = True # assume true, can be changed later
        self.new_game_started: bool = False # assume false, will be overwritten if need be
        # we also have the following:
        # startTime, stopTime (float)
        # team1, team2, current_team (Team)
        # self.sound
        # sound is a part of the class so I can reference it in different functions
        # token_filename
        
        # current_category (str), current_question_points (int)
        # name_to_number (List[int])
        # selected_category_names (List[str])
        # category_index (int)
        # question, correct_answer (str), incorrect_answers (List[str])
        # 
        # stuff pertaining to the GUI
        # cols (int), buttons (List), width (int)
        # team1_score_label, team2_score_label, team1_turn_label, team2_turn_label (Label)
        
        # start timer for debug
        self.startTime = time()
        self.initPopup()
    
    # pre-condition: 
    # post-condition: 
    def checkPreviousWinner(self) -> str:
        """_summary_"""
        
        self.logger.log("Entering checkPreviousWinner...")
        
        # get a list of all log files
        log_files = glob("assets/logs/log_*")
        if not log_files:
            self.logger.log("No previous logs found.",for_debug=False)
            self.logger.log("Exiting checkPreviousWinner.")
            return ""

        # get the SECOND most recent log (most recent is the one generated by the current run)
        #if len(log_files) < 2:
        #    return ""
        log_files.sort(key=getctime, reverse=True)
        #latest_log_file = max(log_files, key=getctime)
        try:
            last_log_file = log_files[1]
        except IndexError as e:
            self.logger.log(f"Last log was not long enough: {e}")
            return ""

        # read the latest log file
        self.logger.log(f"Reading from file '{last_log_file}'",for_debug=False)
        try:
            with open(last_log_file, 'r') as file:
                lines = file.readlines()
        except Exception as e:
            self.logger.log(f"Error reading log file: {e}",output=stderr)
            self.logger.log("Exiting checkPreviousWinner.")
            return ""

        # if the log file is empty
        if not lines:
            self.logger.log("No log entries found.",for_debug=False)
            self.logger.log("Exiting checkPreviousWinner.")
            return ""
        # get the line that contains the winner
        try:
            last_line = lines[-15].strip()
        except IndexError as e: # a game was not finished
            self.logger.log(f"Error: {e}",output=stderr)
            self.logger.log("Exiting checkPreviousWinner.")
            return ""
        
        # is the team name denoted in the log entry?
        win_match = match(r"Game ended\. \(team(1|2)\) (.*) wins!", last_line)
        # we can't do: `if "wins!" in last_line` or something similar because a user could have a
        # team name of the same string which would cause it to not work as intended. Using a regex
        # means we can determine if the entry matches the regex and surgically take the relevant
        # portion (team# or tie) from the log entry

        # note that the fourth-to-last entry should be: Game ended. (team#) team wins! OR
        #                                               Game ended. (tie) It's a tie!
        if win_match:
            self.logger.log(f"Previous winner found to be team{win_match.group(1)} ({win_match.group(2)})",for_debug=False)
            self.logger.log("Exiting checkPreviousWinner.")
            return f"team{win_match.group(1)}"
        elif match(r"Game ended\. \(tie\) It's a tie!", last_line):
            self.logger.log("Previous game outcome found to be a tie",for_debug=False)
            self.logger.log("Exiting checkPreviousWinner.")
            return "tie"
        else:
            self.logger.log("No valid winner or outcome found in the log.")
            self.logger.log("Exiting checkPreviousWinner.")
            return ""
    
    # pre-condition: 
    # post-condition: 
    def initPopup(self, filename: Optional[str] = None) -> None:
        self.logger.log("Entering initPopup...")
        # todo: stop after mainWindow is called
        self.playNoise(is_intro=True)
        
        # init teams popup
        # we need to first create the widgets, then send it to a popup
        # this is the 'top'/outer layout, it should have 1 columns
        team_init_popup = GridLayout(cols=1,spacing=10,padding=10)
        
        team_init_popup.add_widget(Label(
            text="Note: Team 1 goes first",color=(1,0,0,1),bold=True,
            italic=True,underline=True,size_hint_y=None, height=10, valign='middle'))
        
        # 
        popup_content = GridLayout(cols=3,spacing=10,padding=[10,10])
        
        # todo add default values for text input if left blank
        # team 1 widgets
        popup_content.add_widget(Label(text="Team 1:", size_hint_y=None, height=30, valign='middle'))
        team_1_input = TextInput(multiline=False, size_hint_y=None, height=30,hint_text="Team 1")
        popup_content.add_widget(team_1_input)
        # previous winner slot
        if self.PREVIOUS_WINNER == 'team1':
            popup_content.add_widget(Label(text="Winner!", size_hint_y=None, height=30, valign='middle', color=(1,.8,0,1)))
        else:
            popup_content.add_widget(Label(text="", size_hint_y=None, height=30, valign='middle'))
            
        # team 2 widgets
        popup_content.add_widget(Label(text="Team 2:", size_hint_y=None, height=30, valign='middle'))
        team_2_input = TextInput(multiline=False, size_hint_y=None, height=30,hint_text="Team 2")
        popup_content.add_widget(team_2_input)
        # previous winner slot
        if self.PREVIOUS_WINNER == 'team2':
            popup_content.add_widget(Label(text="Winner!", size_hint_y=None, height=30, valign='middle', color=(1,.8,0,1)))
        else:
            popup_content.add_widget(Label(text="", size_hint_y=None, height=30, valign='middle'))
            
        # optional filename
        popup_content.add_widget(Label(text="Filename:",size_hint_y=None,height=30,valign="middle"))
        if filename is None:
            filename_input = TextInput(multiline=False,size_hint_y=None,height=30,hint_text="token")
        else:
            filename_input = TextInput(multiline=False,size_hint_y=None,height=30,text=filename)
        
        popup_content.add_widget(filename_input)
        
        # add to main widget
        team_init_popup.add_widget(popup_content)
        
        # this is all button stuff
        button_container = GridLayout(cols=1,size_hint_y=None, height=30,spacing=10)
        submit_button = Button(text="Done",background_color=(0,1,0,1),size_hint_x=None)
        submit_button.bind(on_press=lambda instance: self.saveTeamNames(team_1_input.text,team_2_input.text,filename_input.text,popup))
        button_container.add_widget(submit_button)
        # finish the right column
        popup_content.add_widget(Label(text="",size_hint_y=None,height=30,valign="middle"))
        
        # final row is it's own grid for spacing
        popup_final_row = GridLayout(cols=3, size_hint_x=1, size_hint_y=None, height=30, spacing=10)
        popup_final_row.add_widget(Label(text="",size_hint_y=None,height=30,valign="middle"))
        popup_final_row.add_widget(button_container)
        # 
        audio_control = GridLayout(cols=2)
        audio_control.add_widget(Label(text="Audio",size_hint_y=None,height=30,valign="middle"))
        audio_checkbox = CheckBox(active=True,size_hint_y=None,height=30)
        audio_checkbox.bind(active=self.changeAudio)
        audio_control.add_widget(audio_checkbox)
        popup_final_row.add_widget(audio_control)
        
        # add to main grid
        team_init_popup.add_widget(popup_final_row)
        
        # add button to popup
        popup = Popup(
            title='Enter team names',
            title_size='20sp',
            auto_dismiss=False,
            content=team_init_popup,
            size_hint=(0.50, 0.55), # x,y
            pos_hint={'center_x': 0.5, 'center_y': 0.5})
        popup.open()
        
        self.logger.log("Exiting initPopup.")
    
    # pre-condition: 
    # post-condition: 
    def changeAudio(self,
        checkbox: CheckBox,
        value: bool
    ) -> None:
        self.use_audio = value # value is True if selected, False if deselected
        self.logger.log(f"Use audio: {self.use_audio}",for_debug=False)
    
    # pre-condition: 
    # post-condition: 
    def saveTeamNames(self,
        team1: str,
        team2: str,
        filename: str,
        popup: Popup
    ) -> None:
        self.logger.log("Entering saveTeamNames...")
        
        # assume that whatever is here is the possible filename
        self.token_filename = filename if filename and exists(filename) else ""
        
        self.team1 = Team(team1)
        self.team2 = Team(team2)
        self.current_team: Team = self.team1
        # todo: ensure team names are not blank
        
        self.logger.log(f"Team 1: {team1}, Team 2: {team2}",for_debug=False)
        popup.dismiss()
        
        if self.token_filename: self.API_TOKEN = JeopardyAPI(self.token_filename)
        else: self.API_TOKEN = JeopardyAPI()
        
        self.logger.log("Exiting saveTeamNames.")
        self.mainWindow()
    
    # pre-condition: 
    # post-condition: 
    def mainWindow(self) -> None:
        self.logger.log("Entering mainWindow...")
        
        # set cols
        self.cols = MAX_CATEGORIES
        self.team1_turn_label = Label(text="^ ^ ^")
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
        self.logger.log(f"Categories: {self.selected_category_names} (indices {selected_category_indices})",for_debug=False)
        
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
                # NOTE: button goes mainWindow -> press -> displayQuestion -> checkAnswer -> removeButton
                button.bind(on_press=lambda instance, btn=button, category=category: self.press(btn,category))
                self.add_widget(button)
                self.buttons.append((button, category))
        
        self.logger.log("Exiting mainWindow.")
    
    # pre-condition: 
    # post-condition: 
    def press(self,
        question_button: Button,
        category: str
    ) -> None:
        self.logger.log("Entering press...")
        
        # add vars to class
        self.current_question_points = int(question_button.text.strip('$')) # monetary value
        self.current_category = category                    # category name
        self.category_index = self.name_to_number[category] # category index
        self.logger.log(f"Pressed {self.current_question_points} in category {category} (index {self.category_index})",for_debug=False)
        
        try: # query api for data
            self.question, self.correct_answer, self.incorrect_answers = self.API_TOKEN.getQuestion(
                category=self.category_index,
                question_amount=self.current_question_points)
        except ValueError as ve:
            #stderr.write(f"Error fetching question: {ve}")
            self.logger.log(f"Error fetching question: {ve}",output=stderr)
            self.logger.log(format_exc())
            return
        except ConnectionError as ce:
            #stderr.write(f"Network error: {ce}")
            self.logger.log(f"Network error: {ce}",output=stderr)
            self.logger.log(format_exc())
            return
        # todo: make sure these exceptions work ok
        
        # just in case the API response did not return everything needed
        if not self.question:
            #stderr.write("Error: Missing question data.")
            self.logger.log("Error: Missing question data.",output=stderr)
            return
        elif not self.correct_answer or not self.incorrect_answers:
            #stderr.write("Error: Missing answer data.")
            self.logger.log("Error: Missing answer data.",output=stderr)
            return
        
        self.logger.log("Exiting press.")
        self.displayQuestion(question_button)
    
    # pre-condition: 
    # post-condition: 
    def displayQuestion(self, question_button: Button) -> None:
        self.logger.log("Entering displayQuestion...")
        self.logger.log(f"Current turn: {self.current_team}",for_debug=False)
        
        all_answers = self.incorrect_answers + [self.correct_answer]
        
        # note the steal opportunity, if it happens
        if self.is_steal:
            print(f"Steal opportunity for {self.current_team.name}!")
            self.logger.log(f"Steal opportunity for {self.current_team.name}!",for_debug=False)
        else: # it is not a steal opportunity, shuffle answers
            #self.is_steal = False
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
        # todo: popup size is off, include category and question amount
        
        self.logger.log("Exiting displayQuestion.")
    
    # pre-condition: 
    # post-condition: 
    def checkAnswer(self,
        selected_answer: str,
        popup: Popup,
        question_button: Button
    ) -> None:
        self.logger.log("Entering checkAnswer...")
        
        is_correct: bool = False
        
        if self.is_steal:
            # dismiss the popup because the current popup is for the old team, and without this
            # there is two popups that the user has to click off of
            popup.dismiss()
            
            if selected_answer == self.correct_answer:
                is_correct = True
                
                # DO NOT swap teams
            else:
                # show that the team is incorrect
                result_popup = Popup(
                    title='Result',
                    content=Label(text="Incorrect."),
                    size_hint=(0.5, 0.5))
                result_popup.open()
                popup.dismiss()
                
                # DO NOT switch teams or re-display the question
                if self.use_audio: kivy_clock.schedule_once(lambda dt: self.playNoise(is_incorrect=True))
                kivy_clock.schedule_once(lambda dt: result_popup.dismiss())
                
                self.removeButton(question_button)
            
            self.is_steal = False
        else: # it is NOT a steal opportunity
            if selected_answer == self.correct_answer:
                is_correct = True  
            else:
                # show that the team is incorrect
                #self.playNoise(is_incorrect=True)
                #popup.dismiss()
                if self.use_audio: kivy_clock.schedule_once(lambda dt: self.playNoise(is_incorrect=True))
                kivy_clock.schedule_once(lambda dt: popup.dismiss())
                
                # 
                self.logger.log(f"{self.current_team} is incorrect, the other team can now steal",for_debug=False)
                
                # switch to next team and mark that it's a steal attempt
                self.is_steal = True
                self.switchTurn()
                self.displayQuestion(question_button)
        
        if is_correct:
            # update score
            self.current_team.updateScore(int(question_button.text.strip('$')))
            # update scores in main window
            self.team1_score_label.text = f"{self.team1.score} pts"
            self.team2_score_label.text = f"{self.team2.score} pts"
            # update UI on next frame
            kivy_clock.schedule_once(lambda dt: self.team1_score_label.canvas.ask_update())
            kivy_clock.schedule_once(lambda dt: self.team2_score_label.canvas.ask_update())
            
            # swap teams for next turn IF IT IS NOT A STEAL
            if not self.is_steal:
                self.switchTurn()
            
            # display result
            result_popup = Popup(
                title='Result',
                content=Label(text="Correct!"),
                size_hint=(0.5, 0.5))
            result_popup.open()
            # close the question popup
            popup.dismiss()
            
            #self.playNoise(is_correct=True)
            # Schedule the sound to play after the result popup is displayed
            if self.use_audio: kivy_clock.schedule_once(lambda dt: self.playNoise(is_correct=True))
            kivy_clock.schedule_once(lambda dt: result_popup.dismiss())
            
            self.removeButton(question_button)
        
        self.logger.log("Exiting checkAnswer.")
    
    # pre-condition: 
    # post-condition: 
    def removeButton(self, question_button: Button) -> None:
        self.logger.log("Entering removeButton...")
        
        # disable button
        question_button.disabled = True
        question_button.background_color = (0.5,0.5,0.5,1) # change color to indicate not valid
        # or hide button
        #question_button.opacity = 0
        #question_button.disabled = True
        
        # remove the button from the list of active buttons
        self.buttons.remove((question_button, self.current_category))
        
        # check if it is the end of game
        if not self.buttons: # if the button list is empty ...
            self.endGame()
        
        #if not self.game_over: self.logger.log("Exiting removeButton.")
        self.logger.log("Exiting removeButton.")
    
    # pre-condition: 
    # post-condition: 
    def switchTurn(self) -> None:
        self.logger.log("Entering switchTurn...")
            
        # swap to the opposite team
        self.current_team = self.team2 if self.current_team == self.team1 else self.team1
        self.logger.log(f"Next turn: {self.current_team}",for_debug=False)

        # update the '^' symbol in the main UI window
        self.team1_turn_label.text, self.team2_turn_label.text = self.team2_turn_label.text, self.team1_turn_label.text
        kivy_clock.schedule_once(lambda dt: self.team1_turn_label.canvas.ask_update())
        kivy_clock.schedule_once(lambda dt: self.team2_turn_label.canvas.ask_update())

        self.logger.log("Exiting switchTurn.")

    # pre-condition: 
    # post-condition: 
    def playNoise(self,
        is_correct: bool = False,
        is_incorrect: bool = False,
        is_intro: bool = False,
        is_outro_1: bool = False,
        is_outro_2: bool = False
    ) -> None:
        self.logger.log("Entering playNoise...")
        
        sound_file: str = "assets/audio/"
        
        if is_correct: sound_file += 'correct.mp3'
        elif is_incorrect: sound_file += 'incorrect.mp3'
        elif is_intro: sound_file += 'intro.mp3'
        elif is_outro_1: sound_file += 'yay_fnaf.mp3'
        elif is_outro_2: sound_file += 'outro.mp3'
        else: # this should not be hit
            self.logger.log("Error: no sound file specified", output=stderr)
            return

        # Load and play the sound
        self.sound = SoundLoader.load(sound_file)
        if self.sound:
            self.logger.log(f"Playing '{sound_file}'...", for_debug=False)
            self.sound.volume = .4 # play at 40% volume
            self.sound.play()
        
        self.logger.log("Exiting playNoise.")

    # pre-condition: full or user-ended game
    # post-condition: 
    def endGame(self) -> None:
        self.logger.log("Entering endGame...")
        
        # check that all question buttons are disabled
        all_used = all(button.disabled for button, _ in self.buttons)
        is_tie: bool = False # assume there is no tie for now
        
        if all_used:
            # all buttons are used, end the game
            self.logger.log("All questions have been used. Ending the game.",for_debug=False)
            self.game_over = True
            
            box = GridLayout(cols=1,spacing=10,padding=10)
            box.add_widget(Label(text=f"Team 1: {self.team1.name} - {self.team1.score} pts"))
            box.add_widget(Label(text=f"Team 2: {self.team2.name} - {self.team2.score} pts"))
            
            # determine winner and SUPER SECRET GARAUNTEED WINNER
            if self.team1.score > self.team2.score or self.team1.name in ["Will","William"]:
                team_num = 1
                winner_text = f"{self.team1.name} wins!"
                self.PREVIOUS_WINNER = "team1" # only if the user wants to play again
            elif self.team2.score > self.team1.score or self.team1.name in ["Will","William"]:
                team_num = 2
                winner_text = f"{self.team2.name} wins!"
                self.PREVIOUS_WINNER = "team2" # only if the user wants to play again
            else:
                winner_text = "It's a tie!"
                is_tie = True
                self.PREVIOUS_WINNER = "tie" # only if the user wants to play again
            
            # what if both team names are guaranteed winners?
            if self.team1.name in ["Will","William"] and self.team2.name in ["Will","William"]:
                winner_text = "It's a tie!"
                is_tie = True
                self.PREVIOUS_WINNER = "tie" # only if the user wants to play again
            
            box.add_widget(Label(text=winner_text, font_size=20, bold=True,color=(0,1,0,1)))
            
            # todo exit button?
            def showPlayAgainButton(dt):
                button_layout = GridLayout(cols=3)
                play_again_button = Button(text="Play again",background_color=(0,1,0,1),size_hint_x=None)
                play_again_button.bind(on_release=lambda instance: self.restartGame())
                for i in range(3):
                    if i == 1: button_layout.add_widget(play_again_button)
                    else: button_layout.add_widget(Label(text="",size_hint_y=None,height=30,valign="middle"))
                box.add_widget(button_layout)
            
            # Display the popup
            end_popup = Popup(
                title="Game Over",
                content=box,
                size_hint=(0.5,0.5), # x,y
                auto_dismiss=False) # we are technically hiding the last question's result
            #end_popup.open()
            kivy_clock.schedule_once(lambda dt: end_popup.open(),0.5)

            if self.use_audio:
                kivy_clock.schedule_once(lambda dt: self.playNoise(is_outro_1=True),1)
                kivy_clock.schedule_once(lambda dt: self.playNoise(is_outro_2=True),5)
                # show button after last outro sfx starts playing
                kivy_clock.schedule_once(showPlayAgainButton, 4.5)
            
            print(self.logger)
            if is_tie: self.logger.log(f"Game ended. (tie) {winner_text}",for_debug=False)
            else: self.logger.log(f"Game ended. (team{team_num}) {winner_text}",for_debug=False)
            
            # log runtime
            self.stopTime = time()
            
            def gameExit(dt):
                if not self.new_game_started: exit(0)
                else:
                    self.logger.log("New game started, exit aborted")
                    self.new_game_started = False
            kivy_clock.schedule_once(gameExit,47) # wait until sound finishes
            self.logger.log(f"Game finished in {self.stopTime-self.startTime:.5f}s")
            
            self.logger.log("Exiting endGame.")
        else: # this should not be hit
            self.logger.log("Game is not yet finished. Questions remaining.")
        
        self.logger.log("Exiting endGame.")
        #return
    
    # pre-condition: full game
    # post-condition: 
    def restartGame(self) -> None:
        self.logger.log("The user is starting a new game",for_debug=False)
        self.new_game_started = True
        # stop the audio
        if self.sound.state == "play":
            self.sound.stop()
        self.sound.unload()
        
        # todo does not work properly
        for widget in self.children[:]:
            self.remove_widget(widget)
        
        self.team1.resetTeam()
        self.team2.resetTeam()
        self.current_team = self.team1
        
        # start timer for debug
        self.startTime = time()
        
        self.initPopup(self.token_filename)


class JeopardyApp(App):
    def build(self):
        return MyLayout()


if __name__ == "__main__":
    JeopardyApp().run()


# todo:
    # - play again button
# 
# timer?