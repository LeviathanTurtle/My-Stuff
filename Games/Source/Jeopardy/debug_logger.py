
# 
# 
# 

from colorama import Fore, Style
from colorama import init as colorama_init
from sys import stdout, stderr
from datetime import datetime
from traceback import format_exc

class DebugLogger:
    # class attribute instead of an instance attribute to avoid making multiple log files
    log_filename: str = ""
    
    def __init__(self, use_color: bool = False) -> None:
        #self.debug = debug
        self.use_color = use_color
        #self.log_entries = []
        if self.use_color: colorama_init(autoreset=True)
        
        # give the log file a name if it does not have one
        if not DebugLogger.log_filename:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            DebugLogger.log_filename = f"log_{timestamp}.txt"

    # pre-condition: 
    # post-condition: 
    def log(self,
        message: str,
        for_debug: bool = True,
        output=stdout
    ) -> None:
        """Logs a message and dumps it."""
        
        if output == stderr: print(message)
        
        if for_debug:
            # make debug output prettier
            tagged_message: str = "[DEBUG] "
            # if the message was for stderr, append tag to log entry
            if output == stderr: tagged_message += "[STDERR] "
            
            tagged_message += message
            
            self.dump(tagged_message,internal_log=True) # add to log file
        
        elif self.use_color:
            # make debug output prettier
            beautified_message: str = f"[{Fore.LIGHTCYAN_EX+Style.BRIGHT}DEBUG{Style.RESET_ALL}] "
            # if the message was for stderr, append tag to log entry
            if output == stderr: beautified_message += f"[{Fore.RED+Style.BRIGHT}STDERR{Style.RESET_ALL}] "
            
            beautified_message += message
            
            self.dump(beautified_message,internal_log=True) # add to log file
        
        else: self.dump(message,internal_log=True) # add to log file
    
    # pre-condition: 
    # post-condition: 
    def dump(self,
        message: str,
        internal_log: bool = False
    ) -> None:
        """Dumps a log message to a file."""

        # Ensure the class-level log_filename is being used
        if not DebugLogger.log_filename:
            # make a timestamp of the format YYYY-MM-DD_HH-MM-SS
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            DebugLogger.log_filename = f"log_{timestamp}.txt"
        
        try:
            with open(f"assets/logs/{self.log_filename}",'a') as file:
                file.write(message + '\n')
        except IOError:
            # file was unable to be opened
            if not internal_log:
                self.log(f"Error: dump to file '{DebugLogger.log_filename}' failed!",output=stderr)
                self.log(format_exc(),internal_log=True)
        except FileNotFoundError:
            # for whatever reason the file DNE and Python could not create it
            if not internal_log:
                self.log(f"Error: '{DebugLogger.log_filename}' not found",output=stderr)
                self.log(format_exc(),internal_log=True)
        except PermissionError: 
            # the file is write-protected, or Python does not have proper permission to access it
            if not internal_log:
                self.log(f"Error: Permission denied to write to '{DebugLogger.log_filename}'", output=stderr)
                self.log(format_exc(),internal_log=True)
        except IsADirectoryError:
            # the filename is a directory
            if not internal_log:
                self.log(f"Error: '{DebugLogger.log_filename}' is a directory, not a file", output=stderr)
                self.log(format_exc(),internal_log=True)
        except OSError as e:
            # corrupted file system or the disk is full
            if not internal_log:
                self.log(f"Error: OS error ({e}) while accessing '{DebugLogger.log_filename}'", output=stderr)
                self.log(format_exc(),internal_log=True)
        except Exception as e:
            # anything else not here already
            if not internal_log:
                self.log(f"Error: An unexpected error occurred ({e}) when accessing {DebugLogger.log_filename}", output=stderr)
                self.log(format_exc(),internal_log=True)
        
        # Are all of these necessary? Probably not, but whatever. I feel it's good practice
        # thinking about and handling edge cases. The following would probably make it easier to
        # read in the code, but less verbose in the output:
        
        #except (IOError, FileNotFoundError, PermissionError, IsADirectoryError, OSError) as e:
        #    if not internal_log:
        #        self.log(f"Error: {e} while writing to 'log.txt'", output=stderr)
        #except Exception as e:
        #    if not internal_log:
        #        self.log(f"Unexpected error: {e}", output=stderr)

    def __str__(self) -> str:
        return f"Log entries located in file 'assets/logs/{DebugLogger.log_filename}'"

