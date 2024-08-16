# This uses the Trivia API from Open Trivia Database (documentation: https://opentdb.com/api_config.php)
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

# ask the user for the amount of jokes they'd like to generate
num_questions = input("How many trivia questions would you like: ")

# query the API
response = get(f'https://opentdb.com/api.php?amount={num_questions}')

# successful response code
if response.status_code == 200:
    # store response in var
    data = response.json()
    
    # for each question in the response
    for i, result in enumerate(data['results']):
        # extract the question and answers from the response
        question = result['question']
        correct_answer = result['correct_answer']
        incorrect_answers = result['incorrect_answers']
        
        # convert any HTML entities to their characters
        question = unescape(question)
        correct_answer = unescape(correct_answer)
        incorrect_answers = [unescape(ans) for ans in incorrect_answers]
        
        print(f"\n  --- Trivia Question {i+1} ---\n{question}")
        
        # display answer choices
        all_answers = incorrect_answers + [correct_answer]
        #print("\nAnswer choices:")
        for j, answer in enumerate(all_answers, 1):
            print(f"{j}. {answer}")
        
        user_answer = input("\n: ")
        
        # checking user's answer
        while not user_answer.isdigit():
            user_answer = input("\nPlease enter a number: ")
            
        if 1 <= int(user_answer) <= len(all_answers):
            if all_answers[int(user_answer) - 1] == correct_answer:
                print("Correct!")
            else:
                print(f"Wrong. The correct answer was: {correct_answer}")
        else:
            print(f"Please enter a number between 1 and {len(all_answers)}.")
# query unsuccessful
else:
    # output error code
    print(f"Error: {response.status_code}\nFailed to retrieve trivia question(s)")
