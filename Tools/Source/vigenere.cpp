/* WORKING WITH A VIGENERE CIPHER
 * William Wadsworth
 * Created: at some point
 * 
 * 
 * [DESCRIPTION]:
 * This program functions as a vigenere encoder/decoder. The user can chose to create a normal
 * vigenere cipher or one using a keyed alphabet, as well as inputting from a file. This program
 * offers printing and dumping functionalities, allowing the user to use the generated cipher
 * elsewhere. The binary was last compiled on 5.24.2024.
 * This was inspired from Lemmino's video, "The Unbreakable Kryptos Code":
 * (https://www.youtube.com/watch?v=jVpsLMCIB0Y) 
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
#include <sstream>      // istringstream
#include <vector>       // used in helper func

using namespace std;
#define ALPHABET_LENGTH 26
#define ALPHABET "abcdefghijklmnopqrstuvwxyz"
bool DEBUG = false;

typedef char VigenereTable[ALPHABET_LENGTH][ALPHABET_LENGTH];


// function to generate a keyed alpbabet based on the keyword provided at runtime
string genKeyedAlphabet(string);
// helper function to determine if a character is in a string
bool isInString(const string&, char);
// helper function to remove duplicate letters in a string
string removeDuplicates(const string&);

// function to create a vigenere table based on a keyed alphabet
VigenereTable* genVigenereTable(const string&);
// function to create a vigenere table
VigenereTable* genVigenereTable();
// function to input a vigenere table from a file
VigenereTable* fileGenVigenereTable(const string&);

// main function to handle user action choices
void action();

// function to dump the current vigenere table to a file
void dumpVigenereTable(const VigenereTable*, const string&);
// helper function to verify a vigenere table has no anomalous values
bool verifyVigenereTable(const VigenereTable*);
// function to print the vigenere table
void printVigenereTable(const VigenereTable*);

// function to encode a word
string encode(const VigenereTable*, string, string);
// function to decode a word
string decode(const VigenereTable*, string, string);
// helper function to remove whitespaces
pair<string, vector<size_t>> removeWhitespaces(const string&);


int main(int argc, char* argv[])
{
    // check CLI arg usage
    if (argc > 2) {
        cerr << "Uasge: ./<exe name> [-d] <token>\nwhere:\n    -d      - optional, enable debug "
             << "output" << endl;
        exit(1);
    }

    // check if -d is present
    if (argc == 2 && !strcmp(argv[1],"-d"))
        DEBUG = true;

    action();

    return 0;
}


/* function to generate a keyed alpbabet based on the keyword provided at runtime
 * pre-condition: keyword must be initialized to a non-empty string of alphabetical characters
 * 
 * post-condition: returns a 26-character string containing all unique letters from the keyword
 *                 parameter followed by the remaining English alphabet characters.
*/
string genKeyedAlphabet(string keyword)
{
    if (DEBUG)
        printf("Entering genKeyedAlphabet...\n");
    
    // remove duplicate letters from keyword
    keyword = removeDuplicates(keyword);

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
        printf("Generated keyed alphabet: %s\nExiting genKeyedAlphabet...\n",keyed_alphabet.c_str());

    return keyed_alphabet;
}


/* helper function to determine if a character is in a string
 * pre-condition: str must be initialized to a non-empty string of alphabetical characters and c
 *                must be initialized to an alphabetical character
 * 
 * post-condition: returns true if the character c is in the string str, otherwise false
*/
bool isInString(const string& str, char c)
{
    if (DEBUG)
        printf("Entering isInString...\n");
    /*
    // for each character in the string
    for (char currentChar : str)
        if (currentChar == c)
            return true; // char in string
    */

    if (DEBUG)
        printf("Exiting isInString...\n");

    // search for first occurence in str
    return str.find(c) != string::npos;

    //return false; // char not in string
}


/* helper function to remove duplicate letters in a string
 * pre-condition: 
 * 
 * post-condition: 
*/
string removeDuplicates(const string& str)
{
    if (DEBUG)
        printf("Entering removeDuplicates...\n");

    string result;

    for (char ch : str)
        // check if the character is already in the new string before appending
        if (result.find(ch) == string::npos)
            result += ch;
    
    if (DEBUG)
        printf("Exiting removeDuplicates...\n");
    
    return result;
}




/* function to create a vigenere table based on a keyed alphabet
 * pre-condition: keyed_alphabet must be initialized to a non-empty string of alphabetical
 *                characters, VigenereTable type must be defined
 * 
 * post-condition: returns a pointer to a newly constructed vigenere table of alphabetical
 *                 characters based on a keyed alphabet
*/
VigenereTable* genVigenereTable(const string& keyed_alphabet)
{
    if (DEBUG)
        printf("Entering genVigenereTable (keyed)...\n");

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
        printVigenereTable(&vigenere_table);
        printf("Exiting genVigenereTable (keyed)...\n");
    }
    
    return &vigenere_table;
}


/* function to create a vigenere table
 * pre-condition: VigenereTable type must be defined
 * 
 * post-condition: returns a pointer to a newly constructed vigenere table of alphabetical
 *                 characters
*/
VigenereTable* genVigenereTable()
{
    if (DEBUG)
        printf("Entering genVigenereTable...\n");

    static VigenereTable vigenere_table;
    
    // fill in the rest of the table
    for (int i=0; i<ALPHABET_LENGTH; i++)
        for (int j=0; j<ALPHABET_LENGTH; j++)
            vigenere_table[i][j] = 'a' + (i+j) % ALPHABET_LENGTH;

    // verify it was generated correctly
    if(!verifyVigenereTable(&vigenere_table))
        cerr << "Warning: generated vigenere table is invalid!\n";

    if (DEBUG) {
        printVigenereTable(&vigenere_table);
        printf("Exiting genVigenereTable...\n");
    }
    
    return &vigenere_table;
}


/* function to input a vigenere table from a file
 * pre-condition: filename string must be initialized to a non-empty string of alphabetical
 *                characters, VigenereTable type must be defined
 * 
 * post-condition: returns a pointer to a newly constructed vigenere table of alphabetical
 *                 characters after reading from a file
*/
VigenereTable* fileGenVigenereTable(const string& filename)
{
    if (DEBUG)
        printf("Entering fileGenVigenereTable...\n");
    
    static VigenereTable vigenere_table;
    // var to hold the read line from input
    string line;

    // open file
    ifstream file (filename);
    // verify file opened
    if (!file) {
        cerr << "Error: file unable to be opened\n";
        exit(2);
    }

    // read input
    for (int i=0; i<ALPHABET_LENGTH; i++)
        // get each line, store in line var
        if (getline(file, line)) {
            // var to extract each char from the line (creates stream from line)
            istringstream iss(line);
            // for each char in the line
            for (int j=0; j<ALPHABET_LENGTH; j++)
                // extract the char from the line and assign to spot in table, report error if fail
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

    // verify it was generated correctly
    if(!verifyVigenereTable(&vigenere_table))
        cerr << "Warning: generated vigenere table is invalid!\n";

    if (DEBUG) {
        printVigenereTable(&vigenere_table);
        printf("Exiting fileGenVigenereTable...\n");
    }

    return &vigenere_table;
}





/* main function to print the vigenere table
 * pre-condition:  
 * 
 * post-condition: nothing is returned, the user selects a function to execute (encode, decode,
 *                 print, etc) or sets boolean finished to true, exiting the function
*/
void action()
{
    if (DEBUG)
        printf("Entering action...\n");

    // 1-1.5. create vigenere table base or from keyword
    // 2. dump vigenere table to file
    // 3. input vigenere table from file
    // 4. print/verify vigenere table
    // 5. encode word
    // 6. decode word
    // 7. exit
    
    // un init: 1, 3, 7
    // init: 2, 4-7

    // bool vars to note if the user specified they are done and if the table is initialized
    bool finished = false, initialized = false;
    // table/matrix var, declared here to avoid re-assigning NULL each action loop
    VigenereTable* vigenere_table = nullptr;

    while (!finished) {
        if (!initialized) {
            cout << "\nWould you like to:\n1. Create a vigenere table from a keyword\n2. Input a "
                    << "vigenere table from a file\n3. Exit program\n\n: ";
            int choice;
            cin >> choice;
            
            switch (choice) {
                // create vigenere table
                case 1:
                {
                    char keyed_resp;
                    cout << "Would you like to use a keyed alphabet for the cipher? [Y/n]: ";
                    cin >> keyed_resp;
                    if (keyed_resp == 'Y') {
                        string keyword, keyed_alphabet;
                        cout << "Enter the keyword to be used for the keyed alphabet: ";
                        cin >> keyword;

                        // create keyed alphabet
                        keyed_alphabet = genKeyedAlphabet(keyword);
                        // generate the vigenere table based on keyed alphabet
                        vigenere_table = genVigenereTable(keyed_alphabet);
                        //vigenere_table = genVigenereTable(genKeyedAlphabet(keyword));
                    } else
                        vigenere_table = genVigenereTable();

                    initialized = true;

                    break;
                }
                // input vigenere table
                case 2:
                {
                    string filename;
                    cout << "Enter the filename containing the vigenere table: ";
                    cin >> filename;

                    vigenere_table = fileGenVigenereTable(filename);
                    initialized = true;

                    break;
                }
                // exit
                case 3:
                    finished = true;
                    break;
                // oopsie, problem
                // if the user can read, this should not be hit
                default:
                    cerr << "Invalid choice. Please try again\n";
                    break;
            }
        }
        // table is initialized
        else {
            cout << "\nWould you like to:\n1. Dump an existing vigenere table to a file\n2. Print the "
                << "generated vigenere table\n3. Encode a word\n4. Decode a word\n5. Exit program\n\n: ";
            int choice;
            cin >> choice;
            
            switch (choice) {
                // dump existing vigenere table
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
                // print/verify
                case 2:
                    if (vigenere_table)
                        printVigenereTable(vigenere_table);
                    else
                        cerr << "Error: vigenere table is not initialized\n";

                    break;
                // encode a word
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
                // decode a word
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
                // exit
                case 5:
                    finished = true;
                    break;
                // oopsie, problem
                // if the user can read, this should not be hit
                default:
                    cerr << "Invalid choice. Please try again\n";
                    break;
            }
        }
    }

    if (DEBUG)
        printf("Exiting action...\n");
}





/* function to dump the current vigenere table to a file
 * pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
 *                alphabetical characters (26x26 char matrix), filename string must be initialized
 *                to a non-empty string of alphabetical characters
 * 
 * post-condition: nothing is returned, the newly constructed file is created and filled with the
 *                 contents of vigenere_table
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
        file << "\n";
    }

    file.close();

    if (DEBUG)
        printf("Exiting dumpVigenereTable...\n");
}


/* helper function to verify a vigenere table has no anomalous values
 * pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
 *                alphabetical characters (26x26 char matrix)
 * 
 * post-condition: true is returned if vigenere_table contains alphabetical characters, otherwise
 *                 false
*/
bool verifyVigenereTable(const VigenereTable* vigenere_table)
{
    if (DEBUG)
        printf("Entering verifyVigenereTable...\n");

    // check that contents of table are a-z or A-Z
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


/* function to print the vigenere table
 * pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
 *                alphabetical characters (26x26 char matrix)
 * 
 * post-condition: if the vigenere table is valid, its contents (in uppercase) are output,
 *                 otherwise only a warning is output
*/
void printVigenereTable(const VigenereTable* table)
{
    if (DEBUG) {
        printf("Entering printVigenereTable...\n");
        printf("Generated table:\n");
    }

    // check that the table is valid
    if (verifyVigenereTable(table)) {
        for (int i=0; i<ALPHABET_LENGTH; i++) {
            for (int j=0; j<ALPHABET_LENGTH; j++)
                cout << (char)toupper((*table)[i][j]) << ' '; // output uppercase letters
            cout << endl;
        }
        cout << "Table is valid\n";
    }
    else
        cout << "Warning: not printing due to table being invalid\n";

    if (DEBUG)
        printf("Exiting printVigenereTable...\n");
}





/* function to encode a word
 * pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
 *                alphabetical characters (26x26 char matrix), plaintext and plaintext_keyword must
 *                be initialized to non-empty strings of alphabetical characters
 * 
 * post-condition: the encoded string is returned, including whitespaces, based on the currently
 *                 loaded vigenere table
*/
string encode(const VigenereTable* vigenere_table, string plaintext, string plaintext_keyword)
{
    if (DEBUG)
        printf("Entering encode...\n");
    
    string ciphertext;

    // remove whitespaces, noting any indices
    auto cleaned_plaintext_result = removeWhitespaces(plaintext);
    // update plaintext
    plaintext = cleaned_plaintext_result.first;
    // store whitespace indices in a vector
    vector<size_t> plaintext_whitespace_indices = cleaned_plaintext_result.second;
    
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
        if (DEBUG)
            cout << "Current plaintext char: " << p_target << ", current keyword char: " << k_target << ".\n";

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

    // re-add whitespaces
    for (size_t index : plaintext_whitespace_indices)
        if (index < ciphertext.length())
            ciphertext.insert(index, 1, ' ');

    if (DEBUG)
        printf("Exiting encode...\n");
    
    return ciphertext;
}


/* function to decode a word
 * pre-condition: VigenereTable type must be defined, vigenere_table must be filled with
 *                alphabetical characters (26x26 char matrix), ciphertext and plaintext_keyword
 *                must be initialized to non-empty strings of alphabetical characters
 * 
 * post-condition: the decoded string is returned, including whitespaces, based on the currently
 *                 loaded vigenere table
*/
string decode(const VigenereTable* vigenere_table, string ciphertext, string plaintext_keyword)
{
    if (DEBUG)
        printf("Entering decode...\n");

    string plaintext;

    // remove whitespaces, noting any indices
    auto cleaned_ciphertext_result = removeWhitespaces(ciphertext);
    // update ciphertext
    ciphertext = cleaned_ciphertext_result.first;
    // store whitespace indices in a vector
    vector<size_t> ciphertext_whitespace_indices = cleaned_ciphertext_result.second;
    
    // first, ensure keystream equals number of chars in ciphertext
    // keystream is less than the ciphertext
    if (plaintext_keyword.length() < ciphertext.length())
        while (plaintext_keyword.length() < ciphertext.length())
            plaintext_keyword += plaintext_keyword.substr(0, ciphertext.length() - plaintext_keyword.length());
    // keystream is greater than the ciphertext
    else if (plaintext_keyword.length() > ciphertext.length())
        plaintext_keyword = plaintext_keyword.substr(0, ciphertext.length());
    
    // the actual decoding bit
    for (size_t i=0; i<ciphertext.length(); i++) {
        char c_target = ciphertext[i];
        char k_target = plaintext_keyword[i];
        if (DEBUG)
            cout << "Current ciphertext char: " << c_target << ", current keyword char: " << k_target << ".\n";

        // find the index for the keystream (x-axis)
        int x_index = -1;
        for (int j=0; j<ALPHABET_LENGTH; j++)
            if ((*vigenere_table)[0][j] == k_target) {
                x_index = j;
                break;
            }
        
        // find the index for the ciphertext (y-axis)
        int y_index = -1;
        for (int j=0; j<ALPHABET_LENGTH; j++)
            if ((*vigenere_table)[j][x_index] == c_target) {
                y_index = j;
                break;
            }

        // ensure both indices were found
        if (y_index == -1) {
            cerr << "Error: character not found in vigenere table." << endl;
            return "";
        }

        plaintext.append(1, (*vigenere_table)[y_index][0]);
    }

    // re-add whitespaces
    for (size_t index : ciphertext_whitespace_indices)
        if (index < plaintext.length())
            plaintext.insert(index, 1, ' ');

    if (DEBUG)
        printf("Exiting decode...\n");
    
    return plaintext;
}


/* helper function to remove whitespaces
 * pre-condition: str must be initialized to a non-empty string of alphabetical characters
 * 
 * post-condition: a pair is returned containing the new string (without whitespaces) and a vector
 *                 of the indices of any occuring whitespaces
*/
pair<string, vector<size_t>> removeWhitespaces(const string& str)
{
    if (DEBUG)
        printf("Entering removeWhitespaces...\n");
        
    // copy var
    string result;
    // vector to store indices of all whitespaces
    vector<size_t> whitespace_indices;

    // for each char in the string
    for (size_t i=0; i<str.length(); i++)
        if (!isspace(str[i])) // if it is not a space, append to string copy
            result += str[i];
        else // it is a whitespace, add its index to the vector
            whitespace_indices.push_back(i);
    
    if (DEBUG)
        printf("Exiting removeWhitespaces...\n");

    // return a pair containing the new string (without whitespaces) and the vector of whitespace
    // indices
    return {result, whitespace_indices};
}


