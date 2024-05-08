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
 * 
 * 3 - invalid operation choice
*/

#include <iostream>
#include <fstream>      // file I/O
#include <map>          // hold config options
using namespace std;


bool DEBUG = false;


//map<string, double> loadFile(const char*);
map<string, double> initMap(ifstream&);
void printMap(const map<string, double>&);
void action(map<string, double>&, const char*);
map<string, double> addMapItem(map<string, double>&, const string&, const double&, const char*);
map<string, double> editMapItem(map<string, double>&, const string&, const double&, const char*);


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

        map<string, double> config = initMap(argv[2]);
    }
    else {
        // not debug

        map<string, double> config = initMap(argv[1]);
    }


    // prompt user for action
    // switch/case


    // 

    return 0;
}


// note: this function only initializes the map
map<string, double> initMap(const char* filename)
{
    if (DEBUG)
        printf("Entering initMap...\n");

    map<string, double> config;

    ifstream file (filename);
    // check file is opened
    if (!file) {
        cerr << "Error: file unable to be opened.\n";
        exit(2);
    }

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
    while (file >> category >> weight)
        config[category] = weight;

    file.close();

    if (DEBUG) {
        printf("Loaded map:\n");
        printMap(config);
        printf("\nExiting initMap...\n");
    }

    return config;
}


void printMap(const map<string, double>& config)
{
    if (DEBUG)
        printf("Entering printMap...\n");

    for (const auto& entry : config)
        //cout << entry.first << ": " << entry.second << "\n";
        printf("%s: %.2f\n", entry.first.c_str(), entry.second);

    if (DEBUG)
        printf("Exiting printMap...\n");
}


void action(map<string, double>& config, const char* filename)
{
    if (DEBUG)
        printf("Entering action...\n");

    printMap(config);

    cout << "Would you like to:\n1. Add an item/weight\n2. Edit an item/weight\n3. Calculate "
         << "final grade\n4. Enter target grade\n\nEnter your choice: ";
    int choice;
    cin >> choice;

    switch(choice) {
        case 1:
        {
            string item;
            cout << "Enter the assignment name: ";
            cin >> item;
            double weight;
            cout << "Enter the assignment's weight: ";
            cin >> weight;

            addMapItem(config, item, weight, filename);
            break;
        }
        case 2:
        {
            printMap(config);
            string item_edit;
            cout << "\nWhat item would you like to edit: ";
            cin >> item_edit;

            break;
        }

        case 3:
            break;

        case 4:
            break;

        // should not be hit if the user can read
        default:
            cerr << "Error: invalid operation choice";
            exit(3);
    }

    if (DEBUG)
        printf("Exiting action...\n");
}


map<string, double> addMapItem(map<string, double>& config, const string& item, const double& weight, const char* filename)
{
    if (DEBUG)
        printf("\nEntering addMapItem...\n");

    config.insert(make_pair(item,weight));

    ofstream file (filename);
    // check file is opened
    if (!file) {
        cerr << "Error: file unable to be opened.\n";
        exit(2);
    }

    file.close();

    if (DEBUG) {
        printf("Updated map:\n");
        printMap(config);
        printf("\nExiting addMapItem...\n");
    }
}


map<string, double> editMapItem(map<string, double>& config, const string& item, const double& weight, const char* filename)
{

}

