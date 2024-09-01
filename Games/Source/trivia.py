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

from argparse import ArgumentParser
from requests import get
from html import unescape
from random import shuffle
from typing import List, Optional, Union, Tuple, Dict


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


# pre-condition: question_data is a dictionary containing the question, correct answer, and
#                incorrect answers, index is a non-negative integer representing the question
#                number
# post-condition: returns a tuple containing the user's answer (as an integer), the correct answer
#                 (as a string), and a list of all possible answers (as strings)
def displayQuestion(
    question_data: Dict[str, str],
    index: int,
    debug: bool = False,
    show_multiple_choice: bool = True
) -> Tuple[Union[str, int], str, List[str]]:
    """Display a trivia question and return user's answer."""
    
    if debug:
        print("Entering displayQuestion...")
    
    # convert any anomalous characters to unicode
    question = unescape(question_data['question'])
    correct_answer = unescape(question_data['correct_answer'])
    incorrect_answers = [unescape(ans) for ans in question_data['incorrect_answers']]

    # we know it's T/F if there is only 1 incorrect answer (2 total)
    is_true_false = len(incorrect_answers) == 1

    # set up answers and output question title
    if is_true_false:
        all_answers = ["True", "False"]
        print(f"\n  --- Trivia Question {index+1} (True/False) ---\n{question}")
    else:
        # combine answers and shuffle
        all_answers = incorrect_answers + [correct_answer]
        shuffle(all_answers)
        print(f"\n  --- Trivia Question {index+1} ---\n{question}")
    
    # display answer choices
    if show_multiple_choice:
        for i, answer in enumerate(all_answers, 1):
            print(f"{i}. {answer}")
    
    # get user answer
    user_answer = input("\n: ")
    
    if is_true_false:
        # checking user's T/F answer
        while not user_answer.lower() in ["true", "false"]:
            user_answer = input("Please enter 'True' or 'False'\n: ")
    elif show_multiple_choice:
        # checking user's multiple choice answer
        while not user_answer.isdigit() and not 1 <= int(user_answer) <= len(all_answers):
            user_answer = input(f"Please enter a number between 1 and {len(all_answers)}\n: ")
        user_answer = int(user_answer)
    
    if debug:
        print("Exiting displayQuestion.")
    return user_answer, correct_answer, all_answers


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



if __name__ == "__main__":
    main()


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