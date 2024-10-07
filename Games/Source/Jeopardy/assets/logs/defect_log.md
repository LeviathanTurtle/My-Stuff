I'm making this after it's mostly finished so there will be no dates. There is also not a particular order.

1. General spacing and layout issues
    - Fixed (repeatedly), usually by adding a blank label
2. Application freezes when audio files are playing
    - Use separate thread for sound effects
3. Outro sound effects still playing after a new game is made
    - Terminate sound thread on new game startup
4. Playsound, pygame.mixer causing issues with Thread and game restart
    - Note that playsound would not stop audio when the game restarted and pygame's mixer.init would 'not be initialized' when the game restarted
    - Used Kivy's SoundLoader instead, no threading
5. Previous winner not being accurately detected
    - Now grabs correct log file, uses a Regex to accurately determine the previous game's outcome
6. Invalid or non-existent filename used for token filename
    - Checks if the file exists before using
7. Current turn label and turn scores not updating after each turn
    - Made the labels part of the class, updates on the next frame after the score is updated
8. If there is no internet, application would freeze when making the API request
    - JeopardyAPI class now checks for internet before continuing
9. The current turn would not always swap after a correct, stolen question
    - Fixed logic issue that was calling turn-swapping function when it should not have
10. Some questions and/or answers would exceed popup or button size
    - Added text wrapping
11. Game would not end after all buttons were used
    - Added list of current buttons, a button is removed from the list after it is used. End game function is called when said list is empty
