
# 
# 
# 

from requests import get                  # make an HTTP GET request
from html import unescape                 # convert anomalous characters to unicode
from typing import Optional, Tuple, List  # variable and function type hinting
from sys import exit
from socket import setdefaulttimeout, socket, AF_INET, SOCK_STREAM, error
from Libraries.debug_logger import DebugLogger

class JeopardyAPI():
    # 
    def __init__(self,
        token_filename: Optional[str] = None,
    ) -> None:
        self.logger = DebugLogger(class_name="JeopardyAPI")
        
        if JeopardyAPI.checkInternet():
            if token_filename: self.getSessionToken(token_filename)
            else: self.getSessionToken()
            
            self.logger.log(f"API token: {self.api_token}")
        else:
            self.logger.log("Error: an internet connection is required!\n",for_stderr=True)
            exit(1)
    
    # pre-condition: none
    # post-condition: returns True if a successful internet connection is made, otherwise False
    @staticmethod
    def checkInternet(
        host: str ="8.8.8.8",
        port: int = 53,
        timeout: int = 3
    ) -> bool:
        """Check if there is internet by attempting to connect to Google's public DNS server."""
        
        try:
            setdefaulttimeout(timeout)
            socket(AF_INET,SOCK_STREAM).connect((host,port))
            return True
        except error:
            return False
            
    # pre-condition: an internet connection
    # post-condition: returns the token generated by the API
    def getSessionToken(self,token_filename: Optional[str] = None) -> str:
        """Generate a new session token for the trivia API."""
        
        self.logger.log("Entering getSessionToken...",for_debug=True)
            
        file_token: Optional[str] = None
        
        # if a filename is provided, read the file for the token
        if token_filename:
            try: # file stuff
                self.logger.log(f"Attempting to read from file '{token_filename}'...")
                with open(token_filename, 'r') as file:
                    file_token = file.read().strip()
            except FileNotFoundError:
                self.logger.log(f"Warning: file '{token_filename}' not found",message_tag="WARNING")
            except IOError:
                self.logger.log(f"Warning: file '{token_filename}' unable to be opened",message_tag="WARNING")
            
            # return the token from the file if it was read
            if file_token:
                self.api_token = file_token # update token
                self.logger.log(f"Successfully read token from file '{token_filename}")
                
                self.logger.log("Exiting getSessionToken.",for_debug=True)
                return self.api_token
            else: 
                self.logger.log(f"Error: token read from file '{token_filename}' is invalid",for_stderr=True)
        
        # otherwise, assume we still need a token, so generate a new one
        self.logger.log("Generating a new token")
        
        response = get("https://opentdb.com/api_token.php?command=request")
        
        if response.status_code == 200:
            data = response.json()
            if data['response_code'] == 0:
                # update token and dump
                self.api_token = data['token']
                self.dumpToken()
                
                self.logger.log("Exiting getSessionToken.",for_debug=True)
                return self.api_token
            else:
                self.logger.log(f"Error in resetting token: {data['response_message']}",for_stderr=True)
                raise ValueError(f"Error in resetting token: {data['response_message']}")
        else:
            self.logger.log(f"HTTP Error: {response.status_code}",for_stderr=True)
            raise ConnectionError(f"HTTP Error: {response.status_code}")
    
    # pre-condition: api_token must be initialized as a string representing the API token
    # post-condition: outputs the API token to a file named 'token'
    def dumpToken(self) -> None:
        """Dumps the API token to an external file."""
        
        self.logger.log("Entering dumpToken...",for_debug=True)
            
        try:
            with open("token",'w') as file:
                file.write(self.api_token)
        except IOError as e:
            self.logger.log(f"Error dumping API token (Error: {e})",for_stderr=True)
            raise e
        
        self.logger.log("Dumped API token to filename 'token'")
        self.logger.log("Exiting dumpToken.",for_debug=True)

    # pre-condition: an internet connection, api_token must be initialized as a string representing the
    #                API token
    # post-condition: returns the re-generated token by the API
    def resetToken(self) -> str:
        """Reset the session token."""
        
        self.logger.log("Entering resetToken...",for_debug=True)
            
        response = get(f"https://opentdb.com/api_token.php?command=reset&token={self.api_token}")
        
        if response.status_code == 200:
            data = response.json()
            if data['response_code'] == 0:
                # update token and dump
                self.api_token = data['token']
                self.dumpToken()
                
                self.logger.log("Exiting resetToken.",for_debug=True)
                return self.api_token
            else:
                message = data['response_message']
                self.logger.log(f"Error in resetting token: {message}",for_stderr=True)
                raise ValueError(f"Error in resetting token: {message}")
        else:
            self.logger.log(f"HTTP Error: {response.status_code}",for_stderr=True)
            raise ConnectionError(f"HTTP Error: {response.status_code}")

    # pre-condition: category must be initialized as a string, question_amount must be initialized to a
    #                number [9,32], API_TOKEN must be initialized to a valid API token
    # post-condition: a tuple containing the question (string), the correct answer (string), and a list
    #                 of three incorrect answers (list of strings)
    def getQuestion(self,
        category: int,
        question_amount: int
    ) -> Tuple[str, str, List[str]]:
        """Fetch a trivia question from the API."""
        
        # basically from trivia.py
        self.logger.log(f"Entering getQuestion... Cat {category} for ${question_amount}",for_debug=True)
        
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
        self.logger.log(f"API URL: {api_url}")
            
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
                    
                    self.logger.log(f"Question: {question} ({correct_answer})")
                    self.logger.log("Exiting getQuestion.",for_debug=True)
                    return question, correct_answer, incorrect_answers
                # reset token if expired or all questions used
                elif data['response_code'] in [3,4]:
                    self.api_token = self.resetToken()
                else:
                    self.logger.log(f"API response error {data['response_code']}\n",for_stderr=True)
                    raise ValueError(f"API response error {data['response_code']}\n")
            else:
                self.logger.log(f"Failed to fetch question from API ({response.status_code})\n",for_stderr=True)
                raise ValueError(f"Failed to fetch question from API ({response.status_code})\n")

    # pre-condition: 
    # post-condition: 
    def __str__(self) -> str:
        return f"API token: {self.api_token}"