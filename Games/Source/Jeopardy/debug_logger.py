
# 
# 
# 

from colorama import Fore, Style
from colorama import init as colorama_init
from datetime import datetime
from traceback import format_exc
from sys import stderr
from os import path, mkdir, makedirs

class DebugLogger:
    # class attribute instead of an instance attribute to avoid making multiple log files
    log_filename: str = ""
    
    def __init__(self,
        use_color: bool = False,
        class_name: str = ""
    ) -> None:
        self.use_color = use_color
        self.class_name = class_name
        
        DebugLogger._make_file()
        
        if self.use_color:
            colorama_init(autoreset=True)
            self.log("Printing logs with colored tags to console.")

    # pre-condition: 
    # post-condition:
    @staticmethod
    def _make_file() -> None:
        # give the log file a name if it does not have one
        if not DebugLogger.log_filename:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            DebugLogger.log_filename = f"log_{timestamp}.txt"
        
        if not (path.exists("./assets/") or path.exists("./assets/logs/")):
            makedirs("./assets/logs/")
    
    # pre-condition: 
    # post-condition: 
    def log(self,
        message: str,
        message_tag: str = "INFO",
        for_debug: bool = False,
        for_stderr: bool = False
    ) -> None:
        """Logs a message with optional color and tags."""
        
        if for_stderr: print(message)
        
        # COLOR CODES:
        # - lightblue: classes
        # - lightyellow: warnings
        # - lightcyan: debug
        # - lightred: errors
        
        if self.use_color:
            # first is the class name
            colored_message: str = f"{Fore.LIGHTBLUE_EX+Style.BRIGHT}[{self.class_name}]{Style.RESET_ALL} " if self.class_name else ""
            
            # second is the type of message being logged
            if for_debug: # include color if specified
                colored_message += f"{Fore.LIGHTCYAN_EX+Style.BRIGHT}[DEBUG]{Style.RESET_ALL} "
            elif for_stderr: # if the message was for stderr, append tag
                colored_message += f"{Fore.LIGHTRED_EX+Style.BRIGHT}[ERROR]{Style.RESET_ALL} "
            else:
                colored_message += f"{Fore.LIGHTYELLOW_EX+Style.BRIGHT if message_tag=='WARNING' else ''}[{message_tag}]{Style.RESET_ALL} "

            colored_message += message
            print(colored_message)
        
        # first is the class name
        tagged_message: str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        tagged_message += f" [{self.class_name}] " if self.class_name else ""
        
        # second is the type of message being logged
        if for_debug: # include color if specified
            tagged_message += "[DEBUG] "
        elif for_stderr: # if the message was for stderr, append tag
            tagged_message += "[ERROR] "
        else:
            tagged_message += f"[{message_tag}] "

        tagged_message += message
        
        try:
            self._dump(tagged_message) # add to log file
        except Exception as e:
            print(f"Failed to log message: {e}",file=stderr)
    
    # pre-condition: 
    # post-condition: 
    def _dump(self, message: str) -> None:
        """Dumps a log message to a file."""

        # ensure the class-level log_filename is being used
        #DebugLogger._make_file()
        
        try:
            with open(f"assets/logs/{self.log_filename}",'a',encoding='utf-8') as file:
                file.write(message + '\n')
        except Exception as e:
            stderr.write(f"Error in DebugLogger with file '{DebugLogger.log_filename}': {e}\n")
            stderr.write(format_exc() + '\n')

    def __str__(self) -> str:
        return f"Log entries located in file 'assets/logs/{DebugLogger.log_filename}'"

