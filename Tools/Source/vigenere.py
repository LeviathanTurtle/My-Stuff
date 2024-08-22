# WORKING WITH A VIGENERE CIPHER
# William Wadsworth
# Created: at some point
# 
# Python-ized 5.27.2024
# Updated 8.17.2024: added optional table creation for standard alphabet, significant
#                    simplification in decode/encode process, PEP 8 Compliance
# 
# 
# [DESCRIPTION]:
# This program functions as a vigenere encoder/decoder. The user can chose to create a normal
# vigenere cipher or one using a keyed alphabet, as well as inputting from a file. This program
# offers printing and dumping functionalities, allowing the user to use the generated cipher
# elsewhere. The binary was last compiled on 5.24.2024.
# This was inspired from Lemmino's video, "The Unbreakable Kryptos Code":
# (https://www.youtube.com/watch?v=jVpsLMCIB0Y)
# 
# 
# [USAGE]:
# To run:  python3 vigenere.py [-d]
# 
# where: 
# [-d]  - optional, enable debug output
# 
# 
# [EXIT/TERMINATING CODES]:
# 0 - program successfully completed a full execution
# 
# 1 - incorrect program arguments used
# 
# 2 - file unable to be opened


# --- IMPORTS ---------------------------------------------------------------------------
from sys import argv, stderr, exit
from typing import Optional, Tuple, List

# --- VIGENERE VARS ---------------------------------------------------------------------
ALPHABET_LENGTH: int = 26
ALPHABET: str = "abcdefghijklmnopqrstuvwxyz"
DEBUG: bool = False

#VigenereTable = [[None for _ in range(ALPHABET_LENGTH)] for _ in range(ALPHABET_LENGTH)]
VigenereTable = List[List[str]]


# --- FUNCTIONS -------------------------------------------------------------------------
# --- GEN KEYED ALPHABET ------------------------
# pre-condition: keyword must be initialized to a non-empty string of alphabetical characters
# post-condition: returns a 26-character string containing all unique letters from the keyword
#                 parameter followed by the remaining English alphabet characters
def gen_keyed_alphabet(keyword: str) -> str:
    """Generates a keyed alphabet by moving the letters of the keyword to the front of the standard
    alphabet and appending the remaining letters."""
    
    if DEBUG:
        print("Entering gen_keyed_alphabet...")
    
    # remove duplicate letters from keyword
    keyword = remove_duplicates(keyword)
    
    # move letters of the keyword to the front of the alphabet
    keyed_alphabet: str = keyword
    
    # append the rest of the letters
    for char in ALPHABET:
        # if the letter is not in the keyword, append
        if char not in keyword:
            keyed_alphabet += char
    
    # check length is ok, should not be hit?
    if len(keyed_alphabet) != ALPHABET_LENGTH:
        print("Warning: keyed alphabet is not 26 characters long!")

    if DEBUG:
        print(f"Generated keyed alphabet: {keyed_alphabet}\nExiting gen_keyed_alphabet.")
    return keyed_alphabet


# --- IS IN STRING ------------------------------
# pre-condition: string must be initialized to a non-empty string of alphabetical characters and c
#                must be initialized to an alphabetical character
# post-condition: returns true if the character c is in the string str, otherwise false
"""
def is_in_string(string: str, c: str) -> bool:
    # search for first occurence in str
    return string.find(c) != -1
"""


# --- REMOVE DUPLICATES -------------------------
# pre-condition: string must be a non-empty string of alphabetical characters
# post-condition: returns a new string without duplicate letters
def remove_duplicates(string: str) -> str:
    """Removes duplicate characters from a string, preserving the order of first occurrence."""
    
    if DEBUG:
        print("Entering remove_duplicates...")
    
    result: str = ""
    for char in string:
        # check if the character is already in the new string before appending
        if char not in result:
            result += char
    
    if DEBUG:
        print("Exiting remove_duplicates.")
    return result


# --- GEN VIGENERE TABLE - KEYED ----------------
# function to create a vigenere table based on a keyed alphabet
# pre-condition: if provided, keyed_alphabet must be initialized to a non-empty string of
#                alphabetical characters, VigenereTable type must be defined
# post-condition: returns a newly constructed vigenere table of alphabetical characters based on a
#                 keyed alphabet
def gen_vigenere_table(keyed_alphabet: Optional[str] = None) -> VigenereTable:
    """Generates a Vigenere table. If a keyed alphabet is provided, it is used as the first row of
    the table; otherwise, the standard alphabet is used."""
    
    if DEBUG:
        print("Entering gen_vigenere_table...")
    
    if keyed_alphabet is None:
        keyed_alphabet = ALPHABET
    
    vigenere_table: VigenereTable = [["" for _ in range(ALPHABET_LENGTH)] for _ in range(ALPHABET_LENGTH)]
    
    # fill in the table
    for i in range(ALPHABET_LENGTH):
        for j in range(ALPHABET_LENGTH):
            vigenere_table[i][j] = keyed_alphabet[(i+j) % ALPHABET_LENGTH]
    
    # verify it was generated correctly
    if not verify_vigenere_table(vigenere_table):
        print("Warning: generated vigenere table is invalid!")
    
    if DEBUG:
        print_vigenere_table(vigenere_table)
        print("Exiting gen_vigenere_table.")
    return vigenere_table


# --- FILE GEN VIGENERE TABLE -------------------
# pre-condition: filename string must be initialized to a non-empty string of alphabetical
#                characters, VigenereTable type must be defined
# post-condition: returns a newly constructed vigenere table of alphabetical characters after
#                 reading from a file
def file_gen_vigenere_table(filename: str) -> Optional[VigenereTable]:
    """Generates a Vigenere table from a file."""
    
    if DEBUG:
        print("Entering file_gen_vigenere_table...")
    
    vigenere_table: VigenereTable = [["" for _ in range(ALPHABET_LENGTH)] for _ in range(ALPHABET_LENGTH)]
    
    try:
        # while the file is open
        with open(filename, 'r') as file:
            # read input
            for i in range(ALPHABET_LENGTH):
                # get each line, store in line var
                line = file.readline().strip()
                
                if not line:
                    stderr.write(f"Error: unable to read line {i+1} from file '{filename}'")
                    if DEBUG:
                        print("Exiting file_gen_vigenere_table.")
                    return None
                # for each char in the line
                for j, char in enumerate(line[:ALPHABET_LENGTH]):
                    # take char from the line and assign to spot in table
                    vigenere_table[i][j] = char
                    if DEBUG:
                        print(f"Current value is {vigenere_table[i][j]} at [{i}][{j}]")
    # report error if file could not be opened
    except IOError:
        stderr.write(f"Error: file '{filename}' unable to be opened")
        if DEBUG:
            print("Exiting file_gen_vigenere_table.")
        return None

    # verify it was generated correctly
    if not verify_vigenere_table(vigenere_table):
        print("Warning: generated vigenere table is invalid!")
    
    if DEBUG:
        print_vigenere_table(vigenere_table)
        print("Exiting file_gen_vigenere_table.")
    return vigenere_table


# --- ACTION ------------------------------------
# pre-condition: VigenereTable type must be defined
# post-condition: nothing is returned, the user selects a function to execute (encode, decode,
#                 print, etc) or sets boolean finished to true, exiting the function
def action() -> None:
    """Main action loop for interacting with the Vigenere table generation, file handling,
    encoding, and decoding."""
    
    if DEBUG:
        print("Entering action...")
    
    # bool vars to note if the user specified they are done and if the table is initialized
    finished: bool = False
    initialized: bool = False
    # table/matrix var, declared here to avoid re-assigning None each action loop
    vigenere_table = Optional[VigenereTable] = None

    while not finished:
        if not initialized:
            choice = int(input("""\nWould you like to:\n
                               1. Create a vigenere table from a keyword\n
                               2. Input a vigenere table from a file\n
                               3. Exit program\n\n: """))

            match choice:
                # create vigenere table
                case 1:
                    use_keyed_alphabet = input("Would you like to use a keyed alphabet for the cipher? [Y/n]: ")
                    
                    if use_keyed_alphabet.lower() == "Y":
                        keyword = input("Enter the keyword to be used for the keyed alphabet: ")
                        # create keyed alphabet
                        keyed_alphabet = gen_keyed_alphabet(keyword)
                        # generate the vigenere table based on keyed alphabet
                        vigenere_table = gen_vigenere_table(keyed_alphabet)
                        #vigenere_table = genVigenereTable(genKeyedAlphabet(keyword))
                    else:
                        vigenere_table = gen_vigenere_table()
                        
                    initialized = True

                # input vigenere table
                case 2:
                    filename = input("Enter the filename containing the vigenere table: ")
                    vigenere_table = file_gen_vigenere_table(filename)
                    initialized = True
                    
                # exit
                case 3:
                    finished = True

                # oopsie, problem
                # if the user can read, this should not be hit
                case _:
                    stderr.write("Invalid choice. Please try again\n")
        
        else:
            choice= int(input("""\nWould you like to:\n
                              1. Dump an existing vigenere table to a file\n
                              2. Print the generated vigenere table\n
                              3. Encode a word\n
                              4. Decode a word\n
                              5. Exit program\n\n: """))
            
            match choice:
                # dump existing vigenere table
                case 1:
                    filename = input("Enter the filename to dump the cipher to: ")
                    
                    if vigenere_table and verify_vigenere_table(vigenere_table):
                        dump_vigenere_table(vigenere_table, filename)
                    else:
                        stderr.write("Error: vigenere table is either invalid or not generated.\n")

                # print/verify
                case 2:
                    if vigenere_table:
                        print_vigenere_table(vigenere_table)
                    else:
                        stderr.write("Error: vigenere table is not initialized")

                # encode
                case 3:
                    plaintext = input("Enter the plaintext you would like to encode: ")
                    plaintext_keyword = input("Enter the keyword used to encode the plaintext: ")
                    if DEBUG:
                        print(f"Plaintext is '{plaintext}' with the keyword {plaintext_keyword}")
                    
                    ciphertext = encode(vigenere_table,plaintext,plaintext_keyword)
                    
                    if ciphertext:
                        print(f"The ciphertext for the word/phrase '{plaintext}' is {ciphertext}")

                # decode
                case 4:
                    ciphertext = input("Enter the ciphertext you would like to decode: ")
                    plaintext_keyword = input("Enter the keyword used to encode the plaintext: ")
                    if DEBUG:
                        print(f"Ciphertext is '{ciphertext}' with the keyword {plaintext_keyword}")
                    
                    plaintext = decode(vigenere_table,ciphertext,plaintext_keyword)
                    
                    if plaintext:
                        print(f"The plaintext for the ciphertext '{ciphertext}' is {plaintext}")

                # exit
                case 5:
                    finished = True

                # oopsie, problem
                # if the user can read, this should not be hit
                case _:
                    stderr.write("Invalid choice. Please try again\n")
    
    if DEBUG:
        print("Exiting action.")


# --- DUMP VIGENERE TABLE -----------------------
# pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
#                alphabetical characters (26x26 char matrix), filename string must be initialized
#                to a non-empty string of alphabetical characters
# post-condition: nothing is returned, the newly constructed file is created and filled with the
#                 contents of vigenere_table
def dump_vigenere_table(vigenere_table: VigenereTable, filename: str) -> None:
    """Dumps the Vigenere table to a file."""
    
    if DEBUG:
        print("Entering dump_vigenere_table...")
    
    try:
        # while the file is open
        with open(filename, 'w') as file:
            # write contents of cipher to file
            for row in vigenere_table:
                file.write(' '.join(row) + "\n")
    # report error if file could not be opened
    except IOError:
        stderr.write(f"Error: file '{filename}' unable to be opened")
        exit(2)
    
    if DEBUG:
        print("Exiting dump_vigenere_table.")


# --- VERIFY VIGENERE TABLE ---------------------
# pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
#                alphabetical characters (26x26 char matrix)
# post-condition: true is returned if vigenere_table contains alphabetical characters, otherwise
#                 false
def verify_vigenere_table(vigenere_table: VigenereTable) -> bool:
    """Checks that all characters in the Vigenere table are alphabetic."""
    
    if DEBUG:
        print("Entering verify_vigenere_table...")
    
    # check that contents of table are a-z or A-Z
    for i in range(ALPHABET_LENGTH):
        for j in range(ALPHABET_LENGTH):
            if not vigenere_table[i][j].isalpha():
                if DEBUG:
                    print(f"Invalid character found: {vigenere_table[i][j]} at [{i}][{j}]")
                return False
    
    if DEBUG:
        print("Exiting verify_vigenere_table.")
    
    return True


# --- PRINT VIGENERE TABLE ----------------------
# pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
#                alphabetical characters (26x26 char matrix)
# post-condition: if the vigenere table is valid, its contents (in uppercase) are output,
#                 otherwise only a warning is output
def print_vigenere_table(vigenere_table: VigenereTable):
    """Prints the Vigenere table."""
    
    if DEBUG:
        print("Entering print_vigenere_table...\nGenerated table:")
    
    # check that the table is valid
    if verify_vigenere_table(vigenere_table):
        for row in vigenere_table:
            print(' '.join(char.upper() for char in row))  # output uppercase letters
        print("Table is valid")
    else:
        print("Warning: not printing due to table being invalid")
    
    if DEBUG:
        print("Exiting print_vigenere_table.")


# --- ENCODE ------------------------------------
# pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
#                alphabetical characters (26x26 char matrix), plaintext and keyword must be
#                initialized to non-empty strings of alphabetical characters
# post-condition: the encoded string is returned, including whitespaces, based on the currently
#                 loaded vigenere table
def encode(vigenere_table: VigenereTable, plaintext: str, keyword: str) -> str:
    """Encodes plaintext using the Vigenere cipher with the given keyword."""
    
    if DEBUG:
        print("Entering encode...")
        
    ciphertext: str = ""
    
    # remove whitespaces, noting any indices
    # update plaintext, store whitespace indices in a list
    plaintext, whitespace_indices = remove_whitespaces(plaintext)
    
    # ensure keyword length matches plaintext length
    keystream = extend_keystream(keyword, len(plaintext))
    
    # for each char in plaintext and keystream
    for p_char, k_char in zip(plaintext, keystream):
        # row index for plaintext char in alphabet
        y_index: int = ALPHABET.index(p_char)
        # column index keystream char in alphabet
        x_index: int = ALPHABET.index(k_char)
        # update the ciphertext
        ciphertext += vigenere_table[y_index][x_index]
    
    # re-add whitespaces
    ciphertext = reinsert_whitespaces(ciphertext, whitespace_indices)
    
    if DEBUG:
        print("Exiting encode.")
    return ciphertext


# --- DECODE ------------------------------------
# pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
#                alphabetical characters (26x26 char matrix), ciphertext and keyword must be
#                initialized to non-empty strings of alphabetical characters
# post-condition: the decoded string is returned, including whitespaces, based on the currently
#                 loaded vigenere table
def decode(vigenere_table: VigenereTable, ciphertext: str, keyword: str) -> str:
    """Decodes ciphertext using the Vigenere cipher with the given keyword."""
    
    if DEBUG:
        print("Entering decode...")
    
    plaintext: str = ""
    
    # remove whitespaces, noting any indices
    # update ciphertext, store whitespace indices in a list
    ciphertext, whitespace_indices = remove_whitespaces(ciphertext)
    
    # ensure keyword length matches plaintext length
    keystream = extend_keystream(keyword, len(ciphertext))
    
    # for each char in ciphertext and keystream
    for c_char, k_char in zip(ciphertext, keystream):
        # find pos of keystream char in alphabet
        x_index: int = ALPHABET.index(k_char)
        # column index for ciphertext char in vigenere table
        y_index: int = vigenere_table[x_index].index(c_char)
        # update the plaintext
        plaintext += ALPHABET[y_index]
        #plaintext += ALPHABET[vigenere_table[ALPHABET.index(k_char)].index(c_char)]
    
    # re-add whitespaces
    plaintext = reinsert_whitespaces(plaintext, whitespace_indices)
        
    if DEBUG:
        print("Exiting decode.")
    return plaintext


# --- EXTEND KEYSTREAM --------------------------
# pre-condition: keystream is a non-empty string, target_length is a positive integer greater than
#                or equal to 0
# post-condition: returns a string with length equal to target_length. If target_length is greater
#                 than the length of keystream, the keystream is repeated until the target length
#                 is reached. If target_length is less than or equal to the length of keystream,
#                 the keystream is truncated to target_length
def extend_keystream(keyword: str, length: int) -> str:
    """Extends or truncates the keyword to match the desired length."""
    
    if DEBUG:
        print("Entering extend_keystream...")
    
    # if the keyword is shorter than the required length
    if len(keyword) < length:
        if DEBUG:
            print("Exiting extend_keystream.")
        # repeat the keyword to meet length
        return (keyword * (length // len(keyword) + 1))[:length]

    if DEBUG:
        print("Exiting extend_keystream.")
    # truncate the word to the required length
    return keyword[:length]


# --- REINSERT WHITESPACES ----------------------
# pre-condition: string is a string without any whitespace characters, whitespace_indices is a list
#                of integers representing positions in the text where whitespaces were originally
#                located, whitespace_indices contains valid indices that do not exceed the length
#                of the final text
# post-condition: returns a new string with whitespace characters inserted at the specified
#                 indices. The length of the returned string is equal to
#                 len(string) + len(whitespace_indices)
def reinsert_whitespaces(string: str, whitespace_indices: List[int]) -> str:
    """Reinserts whitespaces into the string at the original indices."""
    
    if DEBUG:
        print("Entering reinsert_whitespaces...")
    
    # convert to list for whitespace insertion
    string_list = list(string)
    # add a space at the marked indices
    for index in whitespace_indices:
        string_list.insert(index, ' ')

    if DEBUG:
        print("Exiting reinsert_whitespaces.")
    # join back together as one string
    return ''.join(string_list)


# --- REMOVE WHITESPACES ------------------------------------
# pre-condition: str must be initialized to a non-empty string of alphabetical characters
# post-condition: a tuple is returned containing the new string (without whitespaces) and a list
#                 of the indices of any occuring whitespaces
def remove_whitespaces(string: str) -> Tuple[str, List[int]]:
    """Removes whitespaces from a string, returning the modified string and the indices of removed
    whitespaces."""
    
    if DEBUG:
        print("Entering remove_whitespaces...")
    
    # copy var
    string_without_spaces: str = ""
    # list to store indices of all whitespaces
    whitespace_indices: List[int] = []
    
    # for each char in the string
    for i, char in enumerate(string):
        if char.isspace():
            # it is a whitespace, add its index to the list
            whitespace_indices.append(i)
        else:
            # if it is not a space, append to string copy
            string_without_spaces += char
    
    if DEBUG:
        print("Exiting remove_whitespaces.")
    
    # return a pair containing the new string (without whitespaces) and the list of whitespace
    # indices
    return string_without_spaces, whitespace_indices


def main():
    # check CLI arg usage
    if len(argv) > 2:
        stderr.write("""Usage: python3 vigenere.py [-d] <token>\nwhere:\n
                         -d      - optional, enable debug output""")
        exit(1)

    # check if -d is present
    if len(argv) == 2:
        if argv[1] == "-d":
            DEBUG = True
        else:
            stderr.write("""Usage: python3 vigenere.py [-d] <token>\nwhere:\n
                         -d      - optional, enable debug output""")
            exit(1)

    action()


if __name__ == "__main__":
    main()