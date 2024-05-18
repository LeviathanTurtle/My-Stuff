/* stuff using a vigenere (cryptography)
 * reference lemmino's "Krptos"
 *
 * [DESCRIPTION]:
 * This program 
 * 
 * [USAGE]:
 * To compile:  g++ vigienere.cpp -Wall -o <exe name>
 * To run:      ./<exe name> [-d] <token>
 * where:
 * [-d]    - optional, enable debug output
 * <token> - keyword used in the vigenere cipher
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
 * 
 * 1 - incorrect program arguments used
*/

#include <iostream>
#include <cstring>      // strcmp
using namespace std;

typedef char VigenereTable[26][26];


// function to generate a keyed alpbabet based on the keyword provided at runtime
string genKeyedAlphabet(const string&);
// function to determine if a character is in a string
bool isInString(const string&, char);
// function to create a vigenere table based on a keyed alphabet
VigenereTable* genVigenereTable(const string&);
// function to print the vigenere table
void printVigenereTable(const VigenereTable&);


bool DEBUG = false;
const string ALPHABET = "abcdefghijklmnopqrstuvwxyz";
const int ALPHABET_LENGTH = ALPHABET.length();
//using VigenereTable = array<array<char, 26>,26>;


int main(int argc, char* argv[])
{
    // check CLI arg usage
    if (argc < 2 || argc > 3) {
        cerr << "Uasge: ./<exe name> [-d] <token>\nwhere:\n    -d      - optional, enable debug "
             << "output\n    <token> - keyword used in the vigenere cipher" << endl;
        exit(1);
    }

    // 
    if (!strcmp(argv[1],"-d")) {
        // debug
        DEBUG = true;

        string keyed_alphabet = genKeyedAlphabet(argv[2]);
        // switch/case like final_grade_calc
        genVigenereTable(keyed_alphabet);
    } else {
        // not debug

        string keyed_alphabet = genKeyedAlphabet(argv[1]);
        // switch/case like final_grade_calc
        genVigenereTable(keyed_alphabet);
    }

    return 0;
}


/* function to generate a keyed alpbabet based on the keyword provided at runtime
 * pre-condition: 
 * 
 * post-condition: 
*/
string genKeyedAlphabet(const string& keyword)
{
    if (DEBUG)
        printf("Entering genKeyedAlphabet...\n");

    // move letters of the keyword to the front of the alphabet
    string keyed_alphabet = keyword;
    // append the rest of the letters
    for(int i=0; i<ALPHABET_LENGTH; i++)
        // if the letter is not in the keyword, append
        if (!isInString(keyword, ALPHABET[i]))
            keyed_alphabet.append(1,ALPHABET[i]);
    
    // check length is ok
    if (int(keyed_alphabet.length()) != ALPHABET_LENGTH)
        cout << "Warning: keyed alphabet is not 26 characters long!\n";

    if (DEBUG)
        //printf("Generated keyed alphabet: %s\nExiting genKeyedAlphabet...\n",keyed_alphabet.c_str());
        cout << "Generated keyed alphabet: " << keyed_alphabet << "\nExiting genKeyedAlphabet...\n";

    return keyed_alphabet;
}


/* function to determine if a character is in a string
 * pre-condition: 
 * 
 * post-condition: 
*/
bool isInString(const string& str, char c)
{
    if (DEBUG)
        printf("Entering isInString...\n");
    
    // for each character in the string
    for (char currentChar : str)
        if (currentChar == c)
            return true; // char in string
    
    if (DEBUG)
        printf("Exiting isInString...\n");

    return false; // char not in string
}


/* function to create a vigenere table based on a keyed alphabet
 * pre-condition: 
 * 
 * post-condition: 
*/
VigenereTable* genVigenereTable(const string& keyed_alphabet)
{
    if (DEBUG)
        printf("Entering genVigenereTable...\n");

    // static is necsssary because without it the var would be destroyed when the function exits.
    // So returning a pointer to the var would result in returning a dangling pointer
    static VigenereTable vigenere_table;
    const int KEYED_ALPHABET_LENGTH = keyed_alphabet.length();

    // set the first row of the table to the keyed alphabet
    for (int i=0; i<KEYED_ALPHABET_LENGTH; i++)
        vigenere_table[0][i] = keyed_alphabet[i];
    
    // fill in the rest of the table
    for (int i=1; i<KEYED_ALPHABET_LENGTH; i++)
        for (int j=0; j<KEYED_ALPHABET_LENGTH; j++)
            vigenere_table[i][j] = keyed_alphabet[(i+j) % KEYED_ALPHABET_LENGTH];
            // THIS LOOKS SICK
            //vigenere_table[i][j] = 'a' + keyed_alphabet[(i+j) % KEYED_ALPHABET_LENGTH];
    // 'a'               : starting char
    // (i+j)             : sum of indices, represents pos of char in alphabet for wrapping
    // % ALPHABET_LENGTH : ensures calculated pos stays within alphabet length bounds

    if (DEBUG) {
        printVigenereTable(vigenere_table);
        printf("Exiting genVigenereTable...\n");
    }
    
    return &vigenere_table;
}


/* function to print the vigenere table
 * pre-condition: 
 * 
 * post-condition: 
*/
void printVigenereTable(const VigenereTable& table)
{
    if (DEBUG) {
        printf("Entering printVigenereTable...\n");
        printf("Generated table:\n");
    }

    for (int i=0; i<ALPHABET_LENGTH; i++) {
        for (int j=0; j<ALPHABET_LENGTH; j++)
            cout << (char)toupper(table[i][j]) << ' ';
        cout << endl;
    }

    if (DEBUG)
        printf("Exiting printVigenereTable...\n");
}

