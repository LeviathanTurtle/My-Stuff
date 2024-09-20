# 
# 
# 

from requests import get                  # make an HTTP GET request
from html import unescape                 # convert anomalous characters to unicode
from typing import Optional, Tuple, List  # variable and function type hinting

class JeopardyAPI():
    # 
    def __init__(self,
        token_filename: Optional[str] = None,
        debug: bool = False
    ) -> None:
        if token_filename is None:
            self.getSessionToken()
        else:
            self.getSessionToken(token_filename)
        if debug:
            print(f"[DEBUG] API token: {self.api_token}")
            
    # pre-condition: an internet connection
    # post-condition: returns the token generated by the API
    def getSessionToken(self,
        token_filename: Optional[str] = None,
        debug: bool = False,
    ) -> str:
        """Generate a new session token for the trivia API."""
        
        if debug:
            print("[DEBUG] Entering getSessionToken...")
            
        file_token: Optional[str] = None
            
        if token_filename is not None:
            try: # file stuff
                if debug:
                    print(f"[DEBUG] Reading from file '{token_filename}'")
                with open(token_filename, 'r') as file:
                    file_token = file.read().strip()
            except FileNotFoundError:
                print(f"Warning: file '{token_filename}' not found. Generating a new token")
            except IOError:
                print(f"Warning: file '{token_filename}' unable to be opened. Generating a new token")
        
        # return the token from the file if it was read
        if file_token:
            # update token and dump
            self.api_token = file_token
            self.dumpToken(debug)
            if debug:
                print("[DEBUG] Exiting getSessionToken.")
            return self.api_token
        
        # otherwise, assume we still need a token, so generate a new one
        response = get("https://opentdb.com/api_token.php?command=request")
        
        if response.status_code == 200:
            data = response.json()
            if data['response_code'] == 0:
                # update token and dump
                self.api_token = data['token']
                self.dumpToken(debug)
                if debug:
                    print("[DEBUG] Exiting getSessionToken.")
                return self.api_token
            else:
                raise ValueError(f"Error in resetting token: {data['response_message']}")
        else:
            raise ConnectionError(f"HTTP Error: {response.status_code}")
    
    # pre-condition: api_token must be initialized as a string representing the API token
    # post-condition: outputs the API token to a file named 'token'
    def dumpToken(self,
        debug: bool = False
    ) -> None:
        """Dumps the API token to an external file."""
        
        if debug:
            print("[DEBUG] Entering dumpToken...")
            
        try:
            with open("token",'w') as file:
                file.write(self.api_token)
                if debug:
                    print("Dumped API token to filename 'token'")
        except IOError as e:
            print(f"Error dumping API token (Error: {e})")
            
        if debug:
            print("[DEBUG] Exiting dumpToken.")

    # pre-condition: an internet connection, api_token must be initialized as a string representing the
    #                API token
    # post-condition: returns the re-generated token by the API
    def resetToken(self,
        debug: bool = False
    ) -> str:
        """Reset the session token."""
        
        if debug:
            print("[DEBUG] Entering resetToken...")
            
        response = get(f"https://opentdb.com/api_token.php?command=reset&token={self.api_token}")
        
        if response.status_code == 200:
            data = response.json()
            if data['response_code'] == 0:
                # update token and dump
                self.api_token = data['token']
                self.dumpToken(debug)
                if debug:
                    print("[DEBUG] Exiting resetToken.")
                return self.api_token
            else:
                raise ValueError(f"Error in resetting token: {data['response_message']}")
        else:
            raise ConnectionError(f"HTTP Error: {response.status_code}")

    # pre-condition: category must be initialized as a string, question_amount must be initialized to a
    #                number [9,32], API_TOKEN must be initialized to a valid API token
    # post-condition: a tuple containing the question (string), the correct answer (string), and a list
    #                 of three incorrect answers (list of strings)
    def getQuestion(self,
        category: int,
        question_amount: int,
        debug: bool = False
    ) -> Tuple[str, str, List[str]]:
        """Fetch a trivia question from the API."""
        
        # basically from trivia.py
        if debug:
            print(f"[DEBUG] Entering getQuestion... Cat {category}, ${question_amount}")
        
        # ensure correct scope
        #global API_TOKEN
        
        # init API URL var
        api_url: str = f'https://opentdb.com/api.php?amount=1&token={self.api_token}&category={category}&type=multiple'
        
        # check question difficulty, update difficulty
        if question_amount == 100:
            api_url += '&difficulty=easy'
        elif question_amount in [200, 300]:
            api_url += '&difficulty=medium'
        elif question_amount in [400, 500]:
            api_url += '&difficulty=hard'
        if debug:
            print(f"[DEBUG] API URL: {api_url}")
            
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
                        print("[DEBUG] Exiting getQuestion.")
                    return question, correct_answer, incorrect_answers
                # reset token if all questions used
                elif data['response_code'] == 4:
                    self.api_token = self.resetToken(debug)
                else:
                    raise ValueError(f"API response error {data['response_code']}\n")
            else:
                raise ValueError(f"Failed to fetch question from API ({response.status_code})\n")
