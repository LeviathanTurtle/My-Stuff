
# 
# 
# 

from colorama import Fore, Style
from colorama import init as colorama_init
from sys import stdout, stderr

class DebugLogger:
    def __init__(self, debug: bool) -> None:
        self.debug = debug
        colorama_init(autoreset=True)

    # pre-condition: 
    # post-condition: 
    def log(self,
        message: str,
        for_debug: bool = True,
        output=stdout
    ) -> None:
        """Logs a message and prints it if debug mode is enabled."""
        
        if output == stderr: print(message)
        
        if for_debug:
            # make debug output prettier
            beautified_message: str = f"[{Fore.LIGHTCYAN_EX+Style.BRIGHT}DEBUG{Style.RESET_ALL}] "
            # if the message was for stderr, append tag to log entry
            if output == stderr: beautified_message += "[STDERR] "
            
            beautified_message += message
            
            # print to terminal if debug mode is on
            if self.debug: print(beautified_message)
            # add to log file
            self.dump(beautified_message)
        else:
            # ensure the stderr messages are only printed once in debug mode
            if self.debug and not output == stderr: print(message)
            self.dump(message) # add to log file
        # todo: ensure dump excludes color, and/or '[DEBUG]'
    
    # pre-condition: 
    # post-condition: 
    # todo: better or worse to have many dumps or one large dump?
    def dump(self, message: str) -> None:
        """Dumps a log message to a file."""
        
        # todo: add timestamp in filename
        try:
            with open("log.txt",'w') as file:
                file.write(message)
        except:
            pass


#logger = DebugLogger(True)
#logger.log("Entering dumpToken...")
#logger.log("Increased...",for_debug=False)
#logger.log("Error: ...",output=stderr)