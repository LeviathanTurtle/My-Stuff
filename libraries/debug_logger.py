
# 
# 
# 

from colorama import Fore, Style
from colorama import init as colorama_init
from datetime import datetime
from traceback import format_exc
from sys import stderr

class DebugLogger:
    # class attribute instead of an instance attribute to avoid making multiple log files
    log_filename: str = ""
    
    def __init__(self,
        use_color: bool = False,
        class_name: str = ""
    ) -> None:
        self.use_color = use_color
        self.class_name = class_name
        
        if self.use_color:
            colorama_init(autoreset=True)
            self.log("Using color in log tags")
        
        DebugLogger._make_file()

    # pre-condition: 
    # post-condition:
    @staticmethod
    def _make_file() -> None:
        # give the log file a name if it does not have one
        if not DebugLogger.log_filename:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            DebugLogger.log_filename = f"log_{timestamp}.txt"
    
    # pre-condition: 
    # post-condition: 
    def log(self,
        message: str,
        message_tag: str = "INFO",
        for_debug: bool = False,
        for_stderr: bool = False
    ) -> None:
        """Logs a message and dumps it."""
        
        if for_stderr: print(message)
        
        # COLOR CODES:
        # - lightblue: classes
        # - lightyellow: warnings
        # - lightcyan: debug
        # - lightred: errors
        
        # first is the class name
        tagged_message: str = f"{Fore.LIGHTBLUE_EX+Style.BRIGHT if self.use_color else ''}[{self.class_name}]{Style.RESET_ALL if self.use_color else ''} " if self.class_name else ""
        # second is the type of message being logged
        tagged_message += f"{Fore.LIGHTYELLOW_EX+Style.BRIGHT if self.use_color and message_tag=="WARNING" else ''}[{message_tag}]{Style.RESET_ALL if self.use_color else ''} "
        
        if for_debug: # include color if specified
            tagged_message += f"{Fore.LIGHTCYAN_EX+Style.BRIGHT if self.use_color else ''}[DEBUG]{Style.RESET_ALL if self.use_color else ''} "
        if for_stderr: # if the message was for stderr, append tag
            tagged_message += f"{Fore.LIGHTRED_EX+Style.BRIGHT if self.use_color else ''}[STDERR]{Style.RESET_ALL if self.use_color else ''} "

        tagged_message += message
        
        self._dump(tagged_message) # add to log file
    
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

