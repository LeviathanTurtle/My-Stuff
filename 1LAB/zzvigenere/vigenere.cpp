/* stuff using a vigenere (cryptography)
 * reference lemmino's "Krptos"
 *
 * 
 * [DESCRIPTION]:
 * This program 
 * 
 * 
 * [USAGE]:
 * To compile:  g++ vigienere.cpp -Wall -o <exe name>
 * To run:      ./<exe name> [-d]
 * 
 * where:
 * [-d]    - optional, enable debug output
 * 
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
#define ALPHABET_LENGTH 26
#define DEFAULT_CHAR '\0'

typedef char VigenereTable[ALPHABET_LENGTH][ALPHABET_LENGTH];


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
bool verifyVigenereTable(const VigenereTable*);

// function to encode a word
string encode(const VigenereTable*, const string&, string);
// function to find the index of a char in a row (helper func)
int findCharIndex(const char*, const char&);

// function to decode a word
string decode(const VigenereTable*, const string&, const string&);


bool DEBUG = false;
const string ALPHABET = "abcdefghijklmnopqrstuvwxyz";


int main(int argc, char* argv[])
{
    // check CLI arg usage
    if (argc > 2) {
        cerr << "Uasge: ./<exe name> [-d] <token>\nwhere:\n    -d      - optional, enable debug "
             << "output\n    <token> - keyword used to generate the keyed alphabet in the vigenere"
             << " cipher" << endl;
        exit(1);
    }

    // 
    if (argc == 2 && !strcmp(argv[1],"-d"))
        // debug
        DEBUG = true;

    bool finished = false;
    while (!finished)
        action(finished);

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
    
    // check length is ok, should not be hit?
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
    //const int KEYED_ALPHABET_LENGTH = keyed_alphabet.length();

    // set the first row of the table to the keyed alphabet
    for (int i=0; i<ALPHABET_LENGTH; i++)
        vigenere_table[0][i] = keyed_alphabet[i];
    
    // fill in the rest of the table
    for (int i=1; i<ALPHABET_LENGTH; i++)
        for (int j=0; j<ALPHABET_LENGTH; j++)
            vigenere_table[i][j] = keyed_alphabet[(i+j) % ALPHABET_LENGTH];
            // THIS LOOKS KINDA COOL
            //vigenere_table[i][j] = 'a' + keyed_alphabet[(i+j) % KEYED_ALPHABET_LENGTH];
    // 'a'               : starting char
    // (i+j)             : sum of indices, represents pos of char in alphabet for wrapping
    // % ALPHABET_LENGTH : ensures calculated pos stays within alphabet length bounds

    // verify it was generated correctly
    if(!verifyVigenereTable(&vigenere_table))
        cerr << "Warning: generated vigenere table is invalid!\n";

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
    // 4. print/verify vigenere table
    // 5. encode word
    // 6. decode word
    // 7. exit

    cout << "Would you like to:\n1. Create a vigenere table from a keyword\n2. Dump an existing "
         << "vigenere table to a file\n3. Input a vigenere table from a file\n4. Print and verify "
         << "the generated vigenere table\n5. Encode a word\n6. Decode a word\n7. Exit program\n\n: ";
    int choice;
    cin >> choice;

    // main vars defined here because the uses vary in the switch/case
    string keyed_alphabet;
    // init to null so we can tell if it has been successfully updated
    VigenereTable table = { { DEFAULT_CHAR } };
    VigenereTable* vigenere_table = &table;

    switch (choice) {
        // create vigenere table
        case 1:
        {
            string keyword;
            cout << "Enter the keyword to be used for the keyed alphabet: ";
            cin >> keyword;

            // create keyed alphabet
            keyed_alphabet = genKeyedAlphabet(keyword);
            // generate the vigenere table based on keyed alphabet
            vigenere_table = genVigenereTable(keyed_alphabet);
            //vigenere_table = genVigenereTable(genKeyedAlphabet(keyword));

            printVigenereTable(vigenere_table);

            break;
        }
        // dump existing vigenere table
        case 2:
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
        // input vigenere table
        case 3:
        {
            string filename;
            cout << "Enter the filename containing the vigenere table: ";
            cin >> filename;

            vigenere_table = inputVigenereTable(filename);
            break;
        }        
        // print/verify
        case 4:
            printVigenereTable(vigenere_table);

            if(verifyVigenereTable(vigenere_table))
                printf("Vigenere table is valid\n");
            else
                printf("Vigenere table is invalid\n");
            
            break;
        
        // encode a word
        case 5:
        {
            string plaintext, plaintext_keyword;
            cout << "Enter the plaintext you would like to encode: ";
            cin >> plaintext;
            cout << "Enter the keyword used to encode the plaintext: ";
            cin >> plaintext_keyword;

            string cipher_text = encode(vigenere_table, plaintext, plaintext_keyword);

            printf("The ciphertext for the word %s is %s\n",plaintext.c_str(),cipher_text.c_str());

            break;
        }
        // decode a word
        case 6:
        {

            break;
        }
        // exit
        case 7:
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
    if (DEBUG)
        printf("Entering dumpVigenereTable...\n");

    // create/open file
    ofstream file (filename);
    // verify file opened
    if (!file) {
        cerr << "Error: file unable to be opened\n";
        exit(2);
    }

    // write contents of cipher to file
    for(int i=0; i<ALPHABET_LENGTH; i++) {
        for(int j=0; j<ALPHABET_LENGTH; j++)
            file << (*vigenere_table)[i][j] << ' ';
        cout << "\n";
    }

    file.close();

    if (DEBUG)
        printf("Exiting dumpVigenereTable...\n");
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
    
    // init to null so we can tell if it has been successfully updated
    VigenereTable table = { { DEFAULT_CHAR } };
    VigenereTable* vigenere_table = &table;

    // open file
    ifstream file (filename);
    // verify file opened
    if (!file) {
        cerr << "Error: file unable to be opened\n";
        exit(2);
    }

    // read input
    for(int i=0; i<ALPHABET_LENGTH; i++)
        for(int j=0; j<ALPHABET_LENGTH; j++)
            file >> vigenere_table[i][j];

    file.close();

    // verify it was generated correctly
    if(!verifyVigenereTable(vigenere_table))
        cerr << "Warning: generated vigenere table is invalid!\n";

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
bool verifyVigenereTable(const VigenereTable* vigenere_table)
{
    if (DEBUG)
        printf("Entering verifyVigenereTable...\n");

    // check that contents of table are a-z or A-Z
    for(int i=0; i<ALPHABET_LENGTH; i++)
        for(int j=0; j<ALPHABET_LENGTH; j++)
            if ( !isalpha((*vigenere_table)[i][j]) )
                return false;

    if (DEBUG)
        printf("Exiting verifyVigenereTable...\n");
    
    return true;
}


/* function to encode a word
 * pre-condition: 
 * 
 * post-condition: 
*/
string encode(const VigenereTable* vigenere_table, const string& plaintext, string plaintext_keyword)
{
    if (DEBUG)
        printf("Entering encode...\n");
    
    string ciphertext;
    
    // first, ensure keystream equals number of chars in plaintext
    // keystream is less than the plaintext
    if (plaintext_keyword.length() < plaintext.length())
        while (plaintext_keyword.length() < plaintext.length())
            plaintext_keyword += plaintext_keyword.substr(0, plaintext.length() - plaintext_keyword.length());
    // keystream is greater than the plaintext
    else if (plaintext_keyword.length() > plaintext.length())
        plaintext_keyword = plaintext_keyword.substr(0, plaintext.length());
    
    // the actual encoding bit
    for (size_t i=0; i<plaintext.length(); i++) {
        char p_target = plaintext[i];
        char k_target = plaintext_keyword[i];

        // find the index for the plaintext (y-axis)
        int y_index = -1;
        for (int j=0; j<ALPHABET_LENGTH; j++)
            if ((*vigenere_table)[j][0] == p_target) {
                y_index = j;
                break;
            }

        // find the index for the keystream (x-axis)
        int x_index = -1;
        for (int j=0; j<ALPHABET_LENGTH; j++)
            if ((*vigenere_table)[0][j] == k_target) {
                x_index = j;
                break;
            }

        // ensure both indices were found
        if (y_index == -1 || x_index == -1) {
            cerr << "Error: character not found in vigenere table." << endl;
            return "";
        }

        ciphertext.append(1, (*vigenere_table)[y_index][x_index]);
    }

    if (DEBUG)
        printf("Exiting encode...\n");
    
    return ciphertext;
}


/* function to find the index of a char in a row (helper func)
 * pre-condition: 
 * 
 * post-condition: 
*/
int findCharIndex(const char* row, const char& target)
{
    if (DEBUG)
        printf("Entering findCharIndex...\n");

    for (int i=0; i<ALPHABET_LENGTH; i++)
        if (row[i] == target)
            return i;

    return -1;

    if (DEBUG)
        printf("Exiting findCharIndex...\n");
}


/* function to decode a word
 * pre-condition: 
 * 
 * post-condition: 
*/
string decode(const VigenereTable* vigenere_table, const string& ciphertext, const string& plaintext_keyword)
{
    if (DEBUG)
        printf("Entering decode...\n");

    return ciphertext;

    if (DEBUG)
        printf("Exiting decode...\n");
}

