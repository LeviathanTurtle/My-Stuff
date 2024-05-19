/* stuff using a vigenere (cryptography)
 * reference lemmino's "Krptos"
 *
 * [DESCRIPTION]:
 * This program 
 * 
 * [USAGE]:
 * To compile:  g++ vigienere.cpp -Wall -o <exe name>
 * To run:      ./<exe name> [-d]
 * where:
 * [-d]    - optional, enable debug output
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
 * 
 * 1 - incorrect program arguments used
 * 
 * 2 - file unable to be opened
*/

#include <iostream>
#include <cstring>      // strcmp
#include <fstream>      // file I/O
using namespace std;

typedef char VigenereTable[26][26];


// function to generate a keyed alpbabet based on the keyword provided at runtime
string genKeyedAlphabet(const string&);
// function to determine if a character is in a string
bool isInString(const string&, char);
// function to create a vigenere table based on a keyed alphabet
VigenereTable* genVigenereTable(const string&);
// function to print the vigenere table
void printVigenereTable(const VigenereTable*);
// function to handle user action choices
void action(bool&);
// function to dump the current vigenere table to a file
void dumpVigenereTable(const VigenereTable*, const string&);
// function to input a vigenere table from a file
VigenereTable* inputVigenereTable(const string&);
// function to verify a vigenere table has no anomalous values
void verifyVigenereTable(const VigenereTable*);


bool DEBUG = false;
const string ALPHABET = "abcdefghijklmnopqrstuvwxyz";
const int ALPHABET_LENGTH = ALPHABET.length();
//using VigenereTable = array<array<char, 26>,26>;


int main(int argc, char* argv[])
{
    // check CLI arg usage
    if (argc < 2 || argc > 3) {
        cerr << "Uasge: ./<exe name> [-d] <token>\nwhere:\n    -d      - optional, enable debug "
             << "output\n    <token> - keyword used to generate the keyed alphabet in the vigenere"
             << " cipher" << endl;
        exit(1);
    }

    // 
    bool finished = false;
    if (!strcmp(argv[1],"-d")) {
        // debug
        DEBUG = true;

        while (!finished)
            action(finished);
    } else {
        // not debug

        while (!finished)
            action(finished);
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
        //printVigenereTable(vigenere_table);
        printf("Exiting genVigenereTable...\n");
    }
    
    return &vigenere_table;
}


/* function to print the vigenere table
 * pre-condition: 
 * 
 * post-condition: 
*/
void printVigenereTable(const VigenereTable* table)
{
    if (DEBUG) {
        printf("Entering printVigenereTable...\n");
        printf("Generated table:\n");
    }

    for (int i=0; i<ALPHABET_LENGTH; i++) {
        for (int j=0; j<ALPHABET_LENGTH; j++)
            cout << (char)toupper((*table)[i][j]) << ' ';
        cout << endl;
    }

    if (DEBUG)
        printf("Exiting printVigenereTable...\n");
}


/* function to print the vigenere table
 * pre-condition: 
 * 
 * post-condition: 
*/
void action(bool& finished)
{
    if (DEBUG)
        printf("Entering action...\n");

    // switch/case, like final_grade_calc

    // 1. create vigenere table from keyword
    // 2. dump vigenere table to file
    // 3. input vigenere table from file
    // 4. encode word
    // 5. decode word
    // 6. exit

    cout << "Would you like to:\n1. Create a vigenere table from a keyword\n2. Dump an existing "
         << "vigenere table to a file\n3. Input a vigenere table from a file\n4. Encode a word\n"
         << "5. Decode a word\n6. Exit program\n\n: ";
    int choice;
    cin >> choice;

    // main vars defined here because the uses vary in the switch/case
    string keyed_alphabet;
    VigenereTable* vigenere_table;

    switch (choice) {
        // create vigenere table
        case 1:
        {
            string keyword;
            cout << "Enter the keyword to be used in the keyed alphabet: ";
            cin >> keyword;

            // create keyed alphabet based on CLI arg
            keyed_alphabet = genKeyedAlphabet(keyword);
            // generate the vigenere table based on keyed alphabet
            vigenere_table = genVigenereTable(keyed_alphabet);

            printVigenereTable(vigenere_table);

            break;
        }
        // dump existing vigenere table
        case 2:
        {
            string filename;
            cout << "Enter the filename to dump the cipher to: ";
            cin >> filename;

            if (vigenere_table != 0)
                dumpVigenereTable(vigenere_table, filename);
            else 
                cerr << "Error: no vigenere table generated.\n";

            break;
        }
        // input vigenere table
        case 3:
        {
            string filename;
            cout << "Enter the filename containing the vigenere table: ";
            cin >> filename;

            vigenere_table = inputVigenereTable(filename);
            break;
        }
        // encode a word
        case 4:
            break;

        // decode a word
        case 5:
            break;

        // exit
        case 6:
            finished = true;
            break;

        // oopsie, problem
        // if the user can read, this should not be hit
        //default:
    }

    if (DEBUG)
        printf("Exiting action...\n");
}


/* function to dump the current vigenere table to a file
 * pre-condition: 
 * 
 * post-condition: 
*/
void dumpVigenereTable(const VigenereTable* vigenere_table, const string& filename)
{

}


/* function to input a vigenere table from a file
 * pre-condition: 
 * 
 * post-condition: 
*/
VigenereTable* inputVigenereTable(const string& filename)
{
    if (DEBUG)
        printf("Entering inputVigenereTable...\n");
    
    VigenereTable* vigenere_table;

    ifstream file (filename);
    if (!file) {
        cerr << "Error: file unable to be opened\n";
        exit(2);
    }

    for(int i=0; i<ALPHABET_LENGTH; i++)
        for(int j=0; j<ALPHABET_LENGTH; j++)
            file >> vigenere_table[i][j];

    
    file.close();

    if (DEBUG) {
        printVigenereTable(vigenere_table);
        printf("Exiting inputVigenereTable...\n");
    }

    return vigenere_table;
}


/* function to verify a vigenere table has no anomalous values
 * pre-condition: 
 * 
 * post-condition: 
*/
void verifyVigenereTable(const VigenereTable* vigenere_table)
{
    if (DEBUG)
        printf("Entering verifyVigenereTable...\n");

    // check that contents of table are a-z or A-Z
    for(int i=0; i<ALPHABET_LENGTH; i++)
        for(int j=0; j<ALPHABET_LENGTH; j++)
            if ( !isalpha((*vigenere_table)[i][j]) ) {
                cerr << "Warning: generated vigenere table is invalid!\n";
                return;
            }

    if (DEBUG)
        printf("Exiting verifyVigenereTable...\n");
}


