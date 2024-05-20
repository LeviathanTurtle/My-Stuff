# WELCOME!

This is a collection of the main programs I have written. Below is a description of what you can expect in each of the main folders. Each folder contains C++ and Python versions, with a few having Rust versions as well (more will be added in the future).



## CALCULATORS
- `coinTotal`: prompts the user to input a price, and the program will calculate and output the minimum amount of coins for each type (quarter, dime, nickel, penny) required to meet the price (Note: as of 5.20.2024, this does not work 100% accurate)

- `factorialGeoseriesCalc`: performs a factorial or geoseries calculation based on a number given from input

- `finalGrade`: calculates a final grade based on 4 labs, 3 quizzes, and one program and test grade. The percentages are fixed, but can be adjusted in the code

- `fourFunction`: 4-function basic calculator

- `gridpointOperations`: prompts the user for two coordinate points, and the operation the user would like to perform. In the context of ellipsoids, this program assumes you are working with a circle

- `investTable`: creates an investment table based on your input. Interest is compounded monthly

- `isMultiple`: takes two integers X and Y passed as CLI arguments and determines if X is a multiple of Y. If X is a multiple of Y, the program will calculate and output each factor until it reaches X

- `isRightTriangle`: takes three sides as input from the user and determines if they make a right triangle

- `moneyCalculator`: prompts the user for the number of 1, 5, 10, 20, 50, and 100 dollar bills they have It then calculates and outputs the total sum of money based on what the user input (Note: as of 5.20.2024, this does not work 100% accurate)



## GAMES
- Hangman
- Tic-tac-toe



## LIBRARIES
A merged collection of the programs in `Calculators` to be used as header files in other programs



## TESTFILES
A collection of test files used when debugging and testing code



## TOOLS
- `compAlgSort`: takes random values from a data file and puts them into an array and calls user-specified sorting algorithms to sort the array from least to greatest. The program outputs the frequency of the swap statements and the number of elements moved

- `create_test`: generates a test file of user-selected datatypes (stored in seaprate TestFiles  directory) where the user specifies the number of values, range of values used, and the datatype. Valid data types are: integers, doubles, floats, characters, and strings. All data types can be used with optional matrix construction except strings (will be fixed in a future update)

- `dataDetails`: loads an array from a file and sorts the data, then outputs the size of the data, median, minimum, maximum. It will then prompt to search the data for a value, and output the number of occurences

- `golfAnalysis`: analyzes a file consisting of golf data (file structure spcified in program). It finds and outputs the player with the lowest number of strokes for the game. The number of players and holes can be different than the number of players and holes in the input file, that can be adjusted at runtime

- `isLeapYear`: prompts the user for their birthday, checks if the date is valid, and displays if the birth year is a leap year

- `letterGradeAssignment`: processes a number of students' test grade based on a data file and the number of students, both provided at runtime. The program then assigns a letter grade to the student based on their (one) test score

- `numberOfFileContents`: reads through the contents of a file and counts how many items are in it

- `personalSales`: calculates and outputs information pertaining to coffee sales based on an input file

- `soccerStats`: will store a soccer player's name, position, number of games played, total goals made, number of shots taken, and total number of minutes played. It will then compute the shot percentage and output the information in a formatted table

- `tempConv`: creates a temperature conversion chart (between Celsius and Kelvin) based on a degree given in Fahrenheit, incrementing by a value imput by the user
