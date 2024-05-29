# WORKING WITH A VIGENERE CIPHER
# William Wadsworth
# Created: at some point
# Python-ized 5.27.2024
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
"""
#include <iostream>
#include <cstring>      // strcmp
#include <fstream>      // file I/O
#include <sstream>      // istringstream
#include <vector>       // used in helper func

using namespace std;
"""
import sys
from typing import Optional, Tuple, List

# --- VIGENERE VARS ---------------------------------------------------------------------
"""
#define ALPHABET_LENGTH 26
#define ALPHABET "abcdefghijklmnopqrstuvwxyz"
bool DEBUG = false;

typedef char VigenereTable[ALPHABET_LENGTH][ALPHABET_LENGTH];
"""
ALPHABET_LENGTH = 26
ALPHABET = "abcdefghijklmnopqrstuvwxyz"

DEBUG = False

#VigenereTable = [[None for _ in range(ALPHABET_LENGTH)] for _ in range(ALPHABET_LENGTH)]
VigenereTable = List[List[str]]

# --- FUNCTIONS -------------------------------------------------------------------------
# --- GEN KEYED ALPHABET ------------------------
"""
string genKeyedAlphabet(string);
string genKeyedAlphabet(string keyword)
{
    if (DEBUG)
        printf("Entering genKeyedAlphabet...\n");
    
    keyword = removeDuplicates(keyword);

    string keyed_alphabet = keyword;

    for(int i=0; i<ALPHABET_LENGTH; i++)
        if (!isInString(keyword, ALPHABET[i]))
            keyed_alphabet.append(1,ALPHABET[i]);
    
    if (int(keyed_alphabet.length()) != ALPHABET_LENGTH)
        cout << "Warning: keyed alphabet is not 26 characters long!\n";

    if (DEBUG)
        printf("Generated keyed alphabet: %s\nExiting genKeyedAlphabet...\n",keyed_alphabet.c_str());

    return keyed_alphabet;
}
"""
# function to generate a keyed alpbabet based on the keyword provided at runtime
# pre-condition: keyword must be initialized to a non-empty string of alphabetical characters
# post-condition: returns a 26-character string containing all unique letters from the keyword
#                 parameter followed by the remaining English alphabet characters
def genKeyedAlphabet(keyword: str) -> str:
    if DEBUG:
        print("Entering genKeyedAlphabet...")
    
    # remove duplicate letters from keyword
    keyword = removeDuplicates(keyword)
    
    # move letters of the keyword to the front of the alphabet
    keyed_alphabet: str = keyword
    
    # append the rest of the letters
    for i in range(ALPHABET_LENGTH):
        # if the letter is not in the keyword, append
        if ALPHABET[i] not in keyword:
            keyed_alphabet += ALPHABET[i]
    
    # check length is ok, should not be hit?
    if len(keyed_alphabet) != ALPHABET_LENGTH:
        print("Warning: keyed alphabet is not 26 characters long!")

    if DEBUG:
        print(f"Generated keyed alphabet: {keyed_alphabet}\nExiting genKeyedAlphabet...")
    
    return keyed_alphabet

# --- IS IN STRING ------------------------------
"""
bool isInString(const string&, char);
bool isInString(const string& str, char c)
{
    if (DEBUG)
        printf("Entering isInString...\n");
        
    if (DEBUG)
        printf("Exiting isInString...\n");

    return str.find(c) != string::npos;
}
"""
# helper function to determine if a character is in a string
# pre-condition: str must be initialized to a non-empty string of alphabetical characters and c
#                must be initialized to an alphabetical character
# post-condition: returns true if the character c is in the string str, otherwise false
def isInString(string: str, c: str) -> bool:
    if DEBUG:
        print("Entering isInString...")
    
    if DEBUG:
        print("Exiting isInString...")
        
    # search for first occurence in str
    return string.find(c) != -1

# --- REMOVE DUPLICATES -------------------------
"""
string removeDuplicates(const string&);
string removeDuplicates(const string& str)
{
    if (DEBUG)
        printf("Entering removeDuplicates...\n");

    string result;

    for (char ch : str)
        if (result.find(ch) == string::npos)
            result += ch;
    
    if (DEBUG)
        printf("Exiting removeDuplicates...\n");
    
    return result;
}
"""
# helper function to remove duplicate letters in a string
# pre-condition: 
# post-condition: 
def removeDuplicates(string: str) -> str:
    if DEBUG:
        print("Entering removeDuplicates...")
    
    result: str = ""
    for char in string:
        # check if the character is already in the new string before appending
        if char not in result:
            result += char
    
    if DEBUG:
        print("Exiting removeDuplicates...")
    
    return result

# --- GEN VIGENERE TABLE - KEYED ----------------
"""
VigenereTable* genVigenereTable(const string&);
VigenereTable* genVigenereTable(const string& keyed_alphabet)
{
    if (DEBUG)
        printf("Entering genVigenereTable (keyed)...\n");

    static VigenereTable vigenere_table;

    for (int i=0; i<ALPHABET_LENGTH; i++)
        vigenere_table[0][i] = keyed_alphabet[i];
    
    for (int i=1; i<ALPHABET_LENGTH; i++)
        for (int j=0; j<ALPHABET_LENGTH; j++)
            vigenere_table[i][j] = keyed_alphabet[(i+j) % ALPHABET_LENGTH];

    if(!verifyVigenereTable(&vigenere_table))
        cerr << "Warning: generated vigenere table is invalid!\n";

    if (DEBUG) {
        printVigenereTable(&vigenere_table);
        printf("Exiting genVigenereTable (keyed)...\n");
    }
    
    return &vigenere_table;
}
"""
# function to create a vigenere table based on a keyed alphabet
# pre-condition: keyed_alphabet must be initialized to a non-empty string of alphabetical
#                characters, VigenereTable type must be defined
# post-condition: returns a newly constructed vigenere table of alphabetical characters based on a
#                 keyed alphabet
def genVigenereTable(keyed_alphabet: str) -> VigenereTable:
    if DEBUG:
        print("Entering genVigenereTable (keyed)...")
    
    vigenere_table: VigenereTable = [['' for _ in range(ALPHABET_LENGTH)] for _ in range(ALPHABET_LENGTH)]
    
    # set the first row of the table to the keyed alphabet
    for i in range(ALPHABET_LENGTH):
        vigenere_table[0][i] = keyed_alphabet[i]
    
    # fill in the rest of the table
    for i in range(1, ALPHABET_LENGTH):
        for j in range(ALPHABET_LENGTH):
            vigenere_table[i][j] = keyed_alphabet[(i+j) % ALPHABET_LENGTH]
    
    # verify it was generated correctly
    if not verifyVigenereTable(vigenere_table):
        print("Warning: generated vigenere table is invalid!")
    
    if DEBUG:
        printVigenereTable(vigenere_table)
        print("Exiting genVigenereTable (keyed)...")
    
    return vigenere_table

# --- GEN VIGENERE TABLE ------------------------
"""
VigenereTable* genVigenereTable();
VigenereTable* genVigenereTable()
{
    if (DEBUG)
        printf("Entering genVigenereTable...\n");

    static VigenereTable vigenere_table;

    for (int i=0; i<ALPHABET_LENGTH; i++)
        for (int j=0; j<ALPHABET_LENGTH; j++)
            vigenere_table[i][j] = 'a' + (i+j) % ALPHABET_LENGTH;

    if(!verifyVigenereTable(&vigenere_table))
        cerr << "Warning: generated vigenere table is invalid!\n";

    if (DEBUG) {
        printVigenereTable(&vigenere_table);
        printf("Exiting genVigenereTable...\n");
    }
    
    return &vigenere_table;
}
"""
# function to create a vigenere table
# pre-condition: VigenereTable type must be defined
# post-condition: returns a newly constructed vigenere table of alphabetical characters
def genVigenereTable() -> VigenereTable:
    if DEBUG:
        print("Entering genVigenereTable...")
    
    vigenere_table: VigenereTable = [['' for _ in range(ALPHABET_LENGTH)] for _ in range(ALPHABET_LENGTH)]

    # fill in the rest of the table
    for i in range(ALPHABET_LENGTH):
        for j in range(ALPHABET_LENGTH):
            vigenere_table[i][j] = 'a' + (i+j) % ALPHABET_LENGTH
    
    # verify it was generated correctly
    if not verifyVigenereTable(vigenere_table):
        print("Warning: generated vigenere table is invalid!")
    
    if DEBUG:
        printVigenereTable(vigenere_table)
        print("Exiting genVigenereTable...")
    
    return vigenere_table

# --- FILE GEN VIGENERE TABLE -------------------
"""
VigenereTable* fileGenVigenereTable(const string&);
VigenereTable* fileGenVigenereTable(const string& filename)
{
    if (DEBUG)
        printf("Entering fileGenVigenereTable...\n");
    
    static VigenereTable vigenere_table;
    string line;

    ifstream file (filename);
    if (!file) {
        cerr << "Error: file unable to be opened\n";
        exit(2);
    }

    for (int i=0; i<ALPHABET_LENGTH; i++)
        if (getline(file, line)) {
            istringstream iss(line);
            
            for (int j=0; j<ALPHABET_LENGTH; j++)
                if (!(iss >> vigenere_table[i][j])) {
                    cerr << "Error: unable to read data from file '" << filename << "'\n";
                    return nullptr;
                } else if (DEBUG)
                    cout << "Current value is " << vigenere_table[i][j] << " at [" << i << "][" << j << "]\n";
        }
        else {
            cerr << "Error: unable to read line " << i+1 << " from file '" << filename << "'\n";
            return nullptr;
        }

    file.close();

    if(!verifyVigenereTable(&vigenere_table))
        cerr << "Warning: generated vigenere table is invalid!\n";

    if (DEBUG) {
        printVigenereTable(&vigenere_table);
        printf("Exiting fileGenVigenereTable...\n");
    }

    return &vigenere_table;
}
"""
# function to input a vigenere table from a file
# pre-condition: filename string must be initialized to a non-empty string of alphabetical
#                characters, VigenereTable type must be defined
# post-condition: returns a newly constructed vigenere table of alphabetical characters after
#                 reading from a file
def fileGenVigenereTable(filename: str) -> VigenereTable:
    if DEBUG:
        print("Entering fileGenVigenereTable...")
    
    vigenere_table: VigenereTable = [['' for _ in range(ALPHABET_LENGTH)] for _ in range(ALPHABET_LENGTH)]
    
    try:
        # while the file is open
        with open(filename, 'r') as file:
            # read input
            for i in range(ALPHABET_LENGTH):
                # get each line, store in line var
                line = file.readline().strip()
                
                if not line:
                    sys.stderr.write(f"Error: unable to read line {i+1} from file '{filename}'")
                    return None
                # for each char in the line
                for j, char in enumerate(line):
                    if j >= ALPHABET_LENGTH:
                        break
                    # take char from the line and assign to spot in table
                    vigenere_table[i][j] = char
                    if DEBUG:
                        print(f"Current value is {vigenere_table[i][j]} at [{i}][{j}]")

    # report error if file could not be opened
    except IOError:
        sys.stderr.write(f"Error: file '{filename}' unable to be opened")
        return None

    # verify it was generated correctly
    if not verifyVigenereTable(vigenere_table):
        print("Warning: generated vigenere table is invalid!")
    
    if DEBUG:
        printVigenereTable(vigenere_table)
        print("Exiting fileGenVigenereTable...")
    
    return vigenere_table

# --- ACTION ------------------------------------
"""
void action();
void action()
{
    if (DEBUG)
        printf("Entering action...\n");

    bool finished = false, initialized = false;
    VigenereTable* vigenere_table = nullptr;

    while (!finished) {
        if (!initialized) {
            cout << "\nWould you like to:\n1. Create a vigenere table from a keyword\n2. Input a "
                    << "vigenere table from a file\n3. Exit program\n\n: ";
            int choice;
            cin >> choice;
            
            switch (choice) {
                case 1:
                {
                    char keyed_resp;
                    cout << "Would you like to use a keyed alphabet for the cipher? [Y/n]: ";
                    cin >> keyed_resp;
                    if (keyed_resp == 'Y') {
                        string keyword, keyed_alphabet;
                        cout << "Enter the keyword to be used for the keyed alphabet: ";
                        cin >> keyword;

                        keyed_alphabet = genKeyedAlphabet(keyword);
                        vigenere_table = genVigenereTable(keyed_alphabet);
                    } else
                        vigenere_table = genVigenereTable();

                    initialized = true;

                    break;
                }
                case 2:
                {
                    string filename;
                    cout << "Enter the filename containing the vigenere table: ";
                    cin >> filename;

                    vigenere_table = fileGenVigenereTable(filename);
                    initialized = true;

                    break;
                }
                case 3:
                    finished = true;
                    break;
                default:
                    cerr << "Invalid choice. Please try again\n";
                    break;
            }
        }
        else {
            cout << "\nWould you like to:\n1. Dump an existing vigenere table to a file\n2. Print the "
                << "generated vigenere table\n3. Encode a word\n4. Decode a word\n5. Exit program\n\n: ";
            int choice;
            cin >> choice;
            
            switch (choice) {
                case 1:
                {
                    string filename;
                    cout << "Enter the filename to dump the cipher to: ";
                    cin >> filename;

                    if (verifyVigenereTable(vigenere_table))
                        dumpVigenereTable(vigenere_table, filename);
                    else 
                        cerr << "Error: vigenere table is either invalid or not generated.\n";

                    break;
                }
                case 2:
                    if (vigenere_table)
                        printVigenereTable(vigenere_table);
                    else
                        cerr << "Error: vigenere table is not initialized\n";

                    break;
                case 3:
                {
                    string plaintext, plaintext_keyword;
                    cout << "Enter the plaintext you would like to encode: ";
                    cin.ignore();
                    getline(cin, plaintext);
                    cout << "Enter the keyword used to encode the plaintext: ";
                    cin >> plaintext_keyword;
                    if (DEBUG)
                        cout << "Plaintext is '" << plaintext << "' with the keyword " << plaintext_keyword << endl;

                    string cipher_text = encode(vigenere_table, plaintext, plaintext_keyword);

                    printf("The ciphertext for the word/phrase '%s' is '%s'\n",plaintext.c_str(),cipher_text.c_str());

                    break;
                }
                case 4:
                {
                    string cipher_text, plaintext_keyword;
                    cout << "Enter the ciphertext you would like to decode: ";
                    cin.ignore();
                    getline(cin, cipher_text);
                    cout << "Enter the keyword used to encode the plaintext: ";
                    cin >> plaintext_keyword;
                    if (DEBUG)
                        cout << "Ciphertext is '" << cipher_text << "' with the keyword " << plaintext_keyword << endl;

                    string plaintext = decode(vigenere_table, cipher_text, plaintext_keyword);

                    printf("The plaintext for the ciphertext '%s' is '%s'\n",cipher_text.c_str(),plaintext.c_str());

                    break;
                }
                case 5:
                    finished = true;
                    break;
                default:
                    cerr << "Invalid choice. Please try again\n";
                    break;
            }
        }
    }

    if (DEBUG)
        printf("Exiting action...\n");
}
"""
# main function to handle user action choices
# pre-condition: 
# post-condition: nothing is returned, the user selects a function to execute (encode, decode,
#                 print, etc) or sets boolean finished to true, exiting the function
def action():
    if DEBUG:
        print("Entering action...")
    
    # bool vars to note if the user specified they are done and if the table is initialized
    finished: bool = False
    initialized: bool = False
    # table/matrix var, declared here to avoid re-assigning None each action loop
    vigenere_table = Optional[VigenereTable] = None

    while not finished:
        if not initialized:
            
            choice: int = int(input("""\nWould you like to:\n1. Create a vigenere table from a 
                                    keyword\n2. Input a vigenere table from a file\n3. Exit
                                    program\n\n: """))

            match choice:
                # create vigenere table
                case 1:
                    keyed_resp: str = input("Would you like to use a keyed alphabet for the cipher? [Y/n]: ")
                    
                    if keyed_resp == "Y":
                        keyword: str = input("Enter the keyword to be used for the keyed alphabet: ")
                        # create keyed alphabet
                        keyed_alphabet: str = genKeyedAlphabet(keyword)
                        # generate the vigenere table based on keyed alphabet
                        vigenere_table: VigenereTable = genVigenereTable(keyed_alphabet)
                        #vigenere_table = genVigenereTable(genKeyedAlphabet(keyword))
                    else:
                        vigenere_table: VigenereTable = genVigenereTable()
                    
                    initialized = True
                    break
                # input vigenere table
                case 2:
                    filename: str = input("Enter the filename containing the vigenere table: ")
                    
                    vigenere_table: VigenereTable = fileGenVigenereTable(filename)
                    initialized = True
                    
                    break
                # exit
                case 3:
                    finished = True
                    break
                # oopsie, problem
                # if the user can read, this should not be hit
                case _:
                    sys.stderr.write("Invalid choice. Please try again\n")
                    break
        
        else:
            choice: int = int(input("""\nWould you like to:\n1. Dump an existing vigenere table to 
                                    a file\n2. Print the generated vigenere table\n3. Encode a
                                    word\n4. Decode a word\n5. Exit program\n\n: """))
            
            match choice:
                # dump existing vigenere table
                case 1:
                    filename: str = input("Enter the filename to dump the cipher to: ")
                    
                    if verifyVigenereTable(vigenere_table):
                        dumpVigenereTable(vigenere_table, filename)
                    else:
                        sys.stderr.write("Error: vigenere table is either invalid or not generated.\n")
                    break
                # print/verify
                case 2:
                    if vigenere_table:
                        printVigenereTable(vigenere_table)
                    else:
                        sys.stderr.write("Error: vigenere table is not initialized")
                    break
                # encode
                case 3:
                    plaintext: str = input("Enter the plaintext you would like to encode: ")
                    plaintext_keyword: str = input("Enter the keyword used to encode the plaintext: ")
                    if DEBUG:
                        print(f"Plaintext is '{plaintext}' with the keyword {plaintext_keyword}")
                    
                    ciphertext: str = encode(vigenere_table,plaintext,plaintext_keyword)
                    
                    if ciphertext == "":
                        print("Warning: not printing ciphertext due to error")
                    else:
                        print(f"The ciphertext for the word/phrase '{plaintext}' is {ciphertext}")
                    break
                # decode
                case 4:
                    ciphertext: str = input("Enter the ciphertext you would like to decode: ")
                    plaintext_keyword: str = input("Enter the keyword used to encode the plaintext: ")
                    if DEBUG:
                        print(f"Ciphertext is '{ciphertext}' with the keyword {plaintext_keyword}")
                    
                    plaintext: str = decode(vigenere_table,ciphertext,plaintext_keyword)
                    
                    if plaintext == "":
                        print("Warning: not printing ciphertext due to error")
                    else:
                        print(f"The plaintext for the ciphertext '{ciphertext}' is {plaintext}")
                    break
                # exit
                case 5:
                    finished = True
                    break
                # oopsie, problem
                # if the user can read, this should not be hit
                case _:
                    sys.stderr.write("Invalid choice. Please try again\n")
                    break
    
    if DEBUG:
        print("Exiting action...")

# --- DUMP VIGENERE TABLE -----------------------
"""
void dumpVigenereTable(const VigenereTable*, const string&);
void dumpVigenereTable(const VigenereTable* vigenere_table, const string& filename)
{
    if (DEBUG)
        printf("Entering dumpVigenereTable...\n");

    ofstream file (filename);
    if (!file) {
        cerr << "Error: file unable to be opened\n";
        exit(2);
    }

    for(int i=0; i<ALPHABET_LENGTH; i++) {
        for(int j=0; j<ALPHABET_LENGTH; j++)
            file << (*vigenere_table)[i][j] << ' ';
        file << "\n";
    }

    file.close();

    if (DEBUG)
        printf("Exiting dumpVigenereTable...\n");
}
"""
# function to dump the current vigenere table to a file
# pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
#                alphabetical characters (26x26 char matrix), filename string must be initialized
#                to a non-empty string of alphabetical characters
# post-condition: nothing is returned, the newly constructed file is created and filled with the
#                 contents of vigenere_table
def dumpVigenereTable(vigenere_table: VigenereTable, filename: str):
    if DEBUG:
        print("Entering dumpVigenereTable...")
    
    try:
        # while the file is open
        with open(filename, 'w') as file:
            # write contents of cipher to file
            for i in range(ALPHABET_LENGTH):
                for j in range(ALPHABET_LENGTH):
                    file.write(vigenere_table[i][j] + ' ')
                file.write("\n")

    # report error if file could not be opened
    except IOError:
        sys.stderr.write(f"Error: file '{filename}' unable to be opened")
        exit(2)
    
    if DEBUG:
        print("Exiting dumpVigenereTable...")

# --- VERIFY VIGENERE TABLE ---------------------
"""
bool verifyVigenereTable(const VigenereTable*);
bool verifyVigenereTable(const VigenereTable* vigenere_table)
{
    if (DEBUG)
        printf("Entering verifyVigenereTable...\n");

    for(int i=0; i<ALPHABET_LENGTH; i++)
        for(int j=0; j<ALPHABET_LENGTH; j++)
            if ( !isalpha((*vigenere_table)[i][j]) ) {
                if (DEBUG)
                    cerr << "Invalid character found: " << (*vigenere_table)[i][j] << " at [" << i << "][" << j << "]\n";
                return false;
            }

    if (DEBUG)
        printf("Exiting verifyVigenereTable...\n");
    
    return true;
}
"""
# helper function to verify a vigenere table has no anomalous values
# pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
#                alphabetical characters (26x26 char matrix)
# post-condition: true is returned if vigenere_table contains alphabetical characters, otherwise
#                 false
def verifyVigenereTable(vigenere_table: VigenereTable) -> bool:
    if DEBUG:
        print("Entering verifyVigenereTable...")
    
    # check that contents of table are a-z or A-Z
    for i in range(ALPHABET_LENGTH):
        for j in range(ALPHABET_LENGTH):
            if not vigenere_table[i][j].isalpha():
                if DEBUG:
                    print(f"Invalid character found: {vigenere_table[i][j]} at [{i}][{j}]")
                return False
    
    if DEBUG:
        print("Exiting verifyVigenereTable...")
    
    return True

# --- PRINT VIGENERE TABLE ----------------------
"""
void printVigenereTable(const VigenereTable*);
void printVigenereTable(const VigenereTable* table)
{
    if (DEBUG) {
        printf("Entering printVigenereTable...\n");
        printf("Generated table:\n");
    }

    if (verifyVigenereTable(table)) {
        for (int i=0; i<ALPHABET_LENGTH; i++) {
            for (int j=0; j<ALPHABET_LENGTH; j++)
                cout << (char)toupper((*table)[i][j]) << ' ';
            cout << endl;
        }
        cout << "Table is valid\n";
    }
    else
        cout << "Warning: not printing due to table being invalid\n";

    if (DEBUG)
        printf("Exiting printVigenereTable...\n");
}
"""
# function to print the vigenere table
# pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
#                alphabetical characters (26x26 char matrix)
# post-condition: if the vigenere table is valid, its contents (in uppercase) are output,
#                 otherwise only a warning is output
def printVigenereTable(vigenere_table: VigenereTable):
    if DEBUG:
        print("Entering printVigenereTable...\nGenerated table:")
    
    # check that the table is valid
    if verifyVigenereTable(vigenere_table):
        for i in range(ALPHABET_LENGTH):
            for j in range(ALPHABET_LENGTH):
                print(vigenere_table[i][j].upper(), end=' ')  # output uppercase letters
            print()
        print("Table is valid")
    else:
        print("Warning: not printing due to table being invalid")
    
    if DEBUG:
        print("Exiting printVigenereTable...")

# --- ENCODE ------------------------------------
"""
string encode(const VigenereTable*, string, string);
string encode(const VigenereTable* vigenere_table, string plaintext, string plaintext_keyword)
{
    if (DEBUG)
        printf("Entering encode...\n");
    
    string ciphertext;

    auto cleaned_plaintext_result = removeWhitespaces(plaintext);
    plaintext = cleaned_plaintext_result.first;
    vector<size_t> plaintext_whitespace_indices = cleaned_plaintext_result.second;
    
    if (plaintext_keyword.length() < plaintext.length())
        while (plaintext_keyword.length() < plaintext.length())
            plaintext_keyword += plaintext_keyword.substr(0, plaintext.length() - plaintext_keyword.length());
    else if (plaintext_keyword.length() > plaintext.length())
        plaintext_keyword = plaintext_keyword.substr(0, plaintext.length());
    
    for (size_t i=0; i<plaintext.length(); i++) {
        char p_target = plaintext[i];
        char k_target = plaintext_keyword[i];
        if (DEBUG)
            cout << "Current plaintext char: " << p_target << ", current keyword char: " << k_target << ".\n";

        int y_index = -1;
        for (int j=0; j<ALPHABET_LENGTH; j++)
            if ((*vigenere_table)[j][0] == p_target) {
                y_index = j;
                break;
            }

        int x_index = -1;
        for (int j=0; j<ALPHABET_LENGTH; j++)
            if ((*vigenere_table)[0][j] == k_target) {
                x_index = j;
                break;
            }

        if (y_index == -1 || x_index == -1) {
            cerr << "Error: character not found in vigenere table." << endl;
            return "";
        }

        ciphertext.append(1, (*vigenere_table)[y_index][x_index]);
    }

    for (size_t index : plaintext_whitespace_indices)
        if (index < plaintext.length())
            ciphertext.insert(index, 1, ' ');

    if (DEBUG)
        printf("Exiting encode...\n");
    
    return ciphertext;
}
"""
# function to encode a word
# pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
#                alphabetical characters (26x26 char matrix), plaintext and plaintext_keyword must
#                be initialized to non-empty strings of alphabetical characters
# post-condition: the encoded string is returned, including whitespaces, based on the currently
#                 loaded vigenere table
def encode(vigenere_table: VigenereTable, plaintext: str, plaintext_keyword: str) -> str:
    if DEBUG:
        print("Entering encode...")
        
    ciphertext: str = ""
    
    # remove whitespaces, noting any indices
    # update plaintext, store whitespace indices in a list
    plaintext, plaintext_whitespace_indices = removeWhitespaces(plaintext)
    
    # first, ensure keystream equals number of chars in plaintext
    # keystream is shorter than the plaintext
    if len(plaintext_keyword) < len(plaintext):
        while len(plaintext_keyword) < len(plaintext):
            plaintext_keyword += plaintext_keyword[:len(plaintext) - len(plaintext_keyword)]
    # keystream is longer than the plaintext
    elif len(plaintext_keyword) > len(plaintext):
        plaintext_keyword = plaintext_keyword[:len(plaintext)]
    
    # the actual encoding bit
    for i in range(len(plaintext)):
        p_target: str = plaintext[i]
        k_target: str = plaintext_keyword[i]
        if DEBUG:
            print(f"Current plaintext char: {p_target}, current keyword char: {k_target}.")
        
        # find the index for the plaintext (y-axis)
        y_index: int = -1
        for j in range(ALPHABET_LENGTH):
            if vigenere_table[j][0] == p_target:
                y_index = j
                break
            
        # find the index for the keystream (x-axis)
        x_index: int = -1
        for j in range(ALPHABET_LENGTH):
            if vigenere_table[0][j] == k_target:
                x_index = j
                break
        
        # ensure both indices were found
        if x_index == -1 or y_index == -1:
            sys.stderr.write("Error: character not found in vigenere table")
            return ""

        plaintext += vigenere_table[y_index][0]
    
    # re-add whitespaces
    for index in plaintext_whitespace_indices:
        if index < len(ciphertext):
            ciphertext = ciphertext[:index] + ' ' + ciphertext[index:]
    
    if DEBUG:
        print("Exiting encode...")
    
    return ciphertext

# --- DECODE ------------------------------------
"""
string decode(const VigenereTable*, string, string);
string decode(const VigenereTable* vigenere_table, string ciphertext, string plaintext_keyword)
{
    if (DEBUG)
        printf("Entering decode...\n");

    string plaintext;

    auto cleaned_ciphertext_result = removeWhitespaces(ciphertext);
    ciphertext = cleaned_ciphertext_result.first;
    vector<size_t> ciphertext_whitespace_indices = cleaned_ciphertext_result.second;
    
    if (plaintext_keyword.length() < ciphertext.length())
        while (plaintext_keyword.length() < ciphertext.length())
            plaintext_keyword += plaintext_keyword.substr(0, ciphertext.length() - plaintext_keyword.length());
    else if (plaintext_keyword.length() > ciphertext.length())
        plaintext_keyword = plaintext_keyword.substr(0, ciphertext.length());
    
    for (size_t i=0; i<ciphertext.length(); i++) {
        char c_target = ciphertext[i];
        char k_target = plaintext_keyword[i];
        if (DEBUG)
            cout << "Current ciphertext char: " << c_target << ", current keyword char: " << k_target << ".\n";

        int x_index = -1;
        for (int j=0; j<ALPHABET_LENGTH; j++)
            if ((*vigenere_table)[0][j] == k_target) {
                x_index = j;
                break;
            }
        
        int y_index = -1;
        for (int j=0; j<ALPHABET_LENGTH; j++)
            if ((*vigenere_table)[j][x_index] == c_target) {
                y_index = j;
                break;
            }

        if (y_index == -1) {
            cerr << "Error: character not found in vigenere table." << endl;
            return "";
        }

        plaintext.append(1, (*vigenere_table)[y_index][0]);
    }

    for (size_t index : ciphertext_whitespace_indices)
        if (index < ciphertext.length())
            plaintext.insert(index, 1, ' ');

    if (DEBUG)
        printf("Exiting decode...\n");
    
    return plaintext;
}
"""
# function to decode a word
# pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
#                alphabetical characters (26x26 char matrix), ciphertext and plaintext_keyword
#                must be initialized to non-empty strings of alphabetical characters
# post-condition: the decoded string is returned, including whitespaces, based on the currently
#                 loaded vigenere table
def decode(vigenere_table: VigenereTable, ciphertext: str, plaintext_keyword: str) -> str:
    if DEBUG:
        print("Entering decode...")
    
    plaintext: str = ""
    
    # remove whitespaces, noting any indices
    # update ciphertext, store whitespace indices in a list
    ciphertext, ciphertext_whitespace_indices = removeWhitespaces(ciphertext)
    
    # first, ensure keystream equals number of chars in ciphertext
    # keystream is shorter than the ciphertext
    if len(plaintext_keyword) < len(ciphertext):
        while len(plaintext_keyword) < len(ciphertext):
            plaintext_keyword += plaintext_keyword[:len(ciphertext) - len(plaintext_keyword)]
    # keystream is longer than the ciphertext
    elif len(plaintext_keyword) > len(ciphertext):
        plaintext_keyword = plaintext_keyword[:len(ciphertext)]
    
    # the actual decoding bit
    for i in range(len(ciphertext)):
        c_target: str = ciphertext[i]
        k_target: str = plaintext_keyword[i]
        if DEBUG:
            print(f"Current ciphertext char: {c_target}, current keyword char: {k_target}.")
        
        # find the index for the keystream (x-axis)
        x_index: int = -1
        for j in range(ALPHABET_LENGTH):
            if vigenere_table[0][j] == k_target:
                x_index = j
                break
        
        # find the index for the ciphertext (y-axis)
        y_index: int = -1
        for j in range(ALPHABET_LENGTH):
            if vigenere_table[j][x_index] == c_target:
                y_index = j
                break
        
        # ensure both indices were found
        if y_index == -1:
            print("Error: character not found in vigenere table")
            return ""

        plaintext += vigenere_table[y_index][0]
    
    # re-add whitespaces
    for index in ciphertext_whitespace_indices:
        if index < len(plaintext):
            plaintext = plaintext[:index] + ' ' + plaintext[index:]
        
    if DEBUG:
        print("Exiting decode...")
    
    return plaintext

# --- REMOVE WHITESPACES ------------------------------------
"""
pair<string, vector<size_t>> removeWhitespaces(const string&);
pair<string, vector<size_t>> removeWhitespaces(const string& str)
{
    if (DEBUG)
        printf("Entering removeWhitespaces...\n");
        
    string result;
    vector<size_t> whitespace_indices;

    for (size_t i=0; i<str.length(); i++)
        if (!isspace(str[i]))
            result += str[i];
        else
            whitespace_indices.push_back(i);
    
    if (DEBUG)
        printf("Exiting removeWhitespaces...\n");

    return {result, whitespace_indices};
}
"""
# helper function to remove whitespaces
# pre-condition: str must be initialized to a non-empty string of alphabetical characters
# post-condition: a tuple is returned containing the new string (without whitespaces) and a list
#                 of the indices of any occuring whitespaces
def removeWhitespaces(string: str) -> Tuple[str, List[int]]:
    if DEBUG:
        print("Entering removeWhitespaces...")
    
    # copy var
    result: str = ""
    # list to store indices of all whitespaces
    whitespace_indices: List[int] = []
    
    # for each char in the string
    for i, char in enumerate(string):
        if not char.isspace():
            # if it is not a space, append to string copy
            result += char
        else:
            # it is a whitespace, add its index to the vector
            whitespace_indices.append(i)
    
    if DEBUG:
        print("Exiting removeWhitespaces...")
    
    # return a pair containing the new string (without whitespaces) and the vector of whitespace
    # indices
    return result, whitespace_indices

# --- MAIN ------------------------------------------------------------------------------
"""
int main(int argc, char* argv[])
{
    if (argc > 2) {
        cerr << "Uasge: ./<exe name> [-d] <token>\nwhere:\n    -d      - optional, enable debug "
             << "output" << endl;
        exit(1);
    }

    if (argc == 2 && !strcmp(argv[1],"-d"))
        DEBUG = true;

    action();

    return 0;
}
"""
# check CLI arg usage
if len(sys.argv) > 2:
    sys.stderr.write("Uasge: ./<exe name> [-d] <token>\nwhere:\n    -d      - optional, enable debug output")
    exit(1)

# check if -d is present
if len(sys.argv) == 2 and sys.argv[1] == "-d":
    DEBUG = True

action()

