/* stuff using a vigienere (cryptography)
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
 * <token> - 
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
 * 
 * 1 - incorrect program arguments used
*/

#include <iostream>
using namespace std;


bool DEBUG = false;


int main(int argc, char* argv[])
{
    // 
    if (argc < 2 || argc > 3) {
        cerr << "Uasge: ./<exe name> [-d] <token>\nwhere:\n    -d - optional, enable debug output"
             << "\n    <token> - " << endl;
        exit(1);
    }

    // 
    if (argv[1] == "-d") {
        // debug
        DEBUG = true;


    } else {
        // not debug
    }

    return 0;
}