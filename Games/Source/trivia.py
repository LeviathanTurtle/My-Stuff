# This uses the Trivia API from Open Trivia Database (documentation: https://opentdb.com/api_config.php)
# 
# Updated 8.16.2024: function decomposition and PEP 8 Compliance
# 
# The base link is: https://opentdb.com/api.php
# where
#     -- REQUIRED --
# amount  : the number of questions to return
# 
#     -- OPTIONAL --
# type     : specify the type of trivia question (MC or T/F)
# encode   : specify the encoding format in API response (urlLegacy, url3986, base64)
# token    : give a session token to ensure new questions are given
# command  : specify token command to be used
# 
# Note about tokens: using a session token allows you to not receive the same question twice. After
# all questions have been received, you can reset the token or get a new one.
# token=...                : use a session token
# command=request          : retrieve a session token
# command=reset&token=...  : reset session token
# 
# Also note the response codes in the API documentation.
# 
# The API allows for other tools as well:
#     - get all categories and IDs: https://opentdb.com/api_category.php
#     - get the number of questions in a category: https://opentdb.com/api_count.php?category=...
#     - get the number of all questions: https://opentdb.com/api_count_global.php

from requests import get
from html import unescape
from typing import List, Optional, Tuple, Dict

DEBUG: bool = False


# pre-condition: num_questions is a positive integer
# post-condition: returns a list of question data if the API call is successful, otherwise None
def fetch_trivia_questions(num_questions: int) -> Optional[List[Dict[str, str]]]:
    """Fetch trivia questions from the OpenTrivia Database API"""
    
    if DEBUG:
        print("Entering fetchTriviaQuestions...")
    
    # query the API
    response = get(f'https://opentdb.com/api.php?amount={num_questions}')
    
    if response.status_code == 200:
        if DEBUG:
            print("Exiting fetchTriviaQuestions.")
        return response.json()['results']
    # query unsuccessful
    else:
        # output error code
        print(f"Error: {response.status_code}\nFailed to retrieve trivia question(s)")
        if DEBUG:
            print("Exiting fetchTriviaQuestions.")
        return None


# pre-condition: question_data is a dictionary containing the question, correct answer, and
#                incorrect answers, index is a non-negative integer representing the question
#                number
# post-condition: returns a tuple containing the user's answer (as an integer), the correct answer
#                 (as a string), and a list of all possible answers (as strings)
def display_question(question_data: Dict[str, str], index: int) -> Tuple[int, str, List[str]]:
    """Display a trivia question and return user's answer."""
    
    if DEBUG:
        print("Entering displayQuestion...")
    
    question = unescape(question_data['question'])
    correct_answer = unescape(question_data['correct_answer'])
    incorrect_answers = [unescape(ans) for ans in question_data['incorrect_answers']]
    
    print(f"\n  --- Trivia Question {index+1} ---\n{question}")
    
    # display answer choices
    all_answers = incorrect_answers + [correct_answer]
    for i, answer in enumerate(all_answers, 1):
        print(f"{i}. {answer}")
    
    user_answer = input("\n: ")
    
    # checking user's answer
    while not user_answer.isdigit() or not (1 <= int(user_answer) <= len(all_answers)):
        user_answer = input(f"Please enter a number between 1 and {len(all_answers)}: ")
    
    if DEBUG:
        print("Exiting displayQuestion.")
    return int(user_answer), correct_answer, all_answers


def main() -> None:
    # ask the user for the amount of jokes they'd like to generate
    num_questions = input("How many trivia questions would you like: ")
    
    if not num_questions.isdigit() or int(num_questions) <= 0:
        print("Please enter a valid positive integer.")
        return

    questions = fetch_trivia_questions(num_questions)

    if questions:
        for i, question_data in enumerate(questions):
            user_answer, correct_answer, all_answers = display_question(question_data, i)
            
            if all_answers[user_answer - 1] == correct_answer:
                print("Correct!")
            else:
                print(f"Wrong. The correct answer was: {correct_answer}")


if __name__ == "__main__":
    main()