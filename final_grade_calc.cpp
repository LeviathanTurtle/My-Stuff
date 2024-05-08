/* FINAL GRADE CALCULATOR
 * William Wadsworth
 * Created: at some point
 * 
 * [DESCRIPTION]:
 * This program calculates the final grade in a class. Assignment names and their weighted
 * percentages are stored in an external file named ______. Default assignments are homework,
 * quizzes, tests, and labs. The user has the option to add additional assignments with their
 * weights or change an existing assignments weighted percentage, which will all be saved in the
 * same config file. 
 * TODO: maybe a table or some kind of optional structured output
 * 
 * [USAGE]:
 * To compile:
 *     g++ final_grade_calc.cpp -Wall -o <exe name>
 * To run:
 *     ./<exe name> [-d] <config file name> 
 * where:
 * [-d] - optional, enable debug output
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed full execution
 * 
 * 1 - invalid CLI args
 * 
 * 2 - file unable to be opened
*/

#include <iostream>
#include <fstream>      // file I/O
#include <map>          // hold config options
using namespace std;


bool DEBUG = false;


void loadFile(const char*);
map<string, double> initMap(ifstream&);
map<string, double> addMapItem(ifstream&);


// the defaults are made up values
//map<string, double> config;


int main(int argc, char* argv[])
{
    // check CLI args
    if(argc != 3) {
        cerr << "Error: invalid arguments. Usage: ./<exe name> [-d] <config file name> \n";
        exit(1);
    }


    // call loadFile func
    // BASED ON CLI ARGS
    if (argv[1] == "-d") {
        // debug
        DEBUG = true;

        loadFile(argv[2]);
    }
    else {
        // not debug

        loadFile(argv[1]);
    }


    // load map


    // prompt user for action
    // switch/case


    // 

    return 0;
}


void loadFile(const char* filename)
{
    if (DEBUG)
        printf("Entering loadFile...\n");
    
    ifstream file (filename);
    // check file is opened
    if (!file) {
        cerr << "Error: file unable to be opened.\n";
        exit(2);
    }



    file.close();

    if (DEBUG)
        printf("Exiting loadFile...\n");
}


// note: this function only initializes the map
map<string, double> initMap(ifstream& file)
{
    if (DEBUG)
        printf("Entering initMap...\n");

    map<string, double> config;

    // defaults
    /*
    config["homework"] = 0.1;
    config["quiz"] = 0.15;
    config["lab"] = 0.15;
    config["project"] = 0.15;
    config["test"] = 0.2;
    config["exam"] = 0.25;
    */
    config.insert(make_pair("homework",0.1));
    config.insert(make_pair("quiz",0.15));
    config.insert(make_pair("lab",0.15));
    config.insert(make_pair("project",0.15));
    config.insert(make_pair("test",0.2));
    config.insert(make_pair("exam",0.25));

    // load from file
    string category;
    double weight;
    //for (int i=0; i<config.size(); i++)
    while (file >> category >> weight)
        config[category] = weight;

    if (DEBUG) {
        for (const auto& entry : config)
            printf("Loaded map:\n%s: %.2f\n", entry.first.c_str(), entry.second);
        printf("\nExiting initMap...\n");
    }
}


map<string, double> addMapItem(ifstream&)
{
    
}