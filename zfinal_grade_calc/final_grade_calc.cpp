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
//#include <algorithm>
//#include <string>
using namespace std;


bool DEBUG = false;


//map<string, double> loadFile(const char*);
map<string, double> initMap(ifstream&);
void printMap(const map<string, double>&);
void action(map<string, double>&, const char*, bool&);
void addMapItem(map<string, double>&, const string&, const double&, const char*);
void editMapItem(map<string, double>&, const string&, const char*);
// calc final grade
double calcFinalGrade(const map<string, double>&);
// enter target grade -- this is for exam
double finalExamCalc(const map<string, double>&, const double&);


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

        // prompt user for action
        // bool var for repeating inputs
        bool finished = false;
        while (!finished)
            action(config, argv[2], finished);
    }
    else {
        // not debug

        map<string, double> config = initMap(argv[1]);

        // prompt user for action
        // bool var for repeating inputs
        bool finished = false;
        while (!finished)
            action(config, argv[1], finished);
    }

    return 0;
}


// note: this function only initializes the map
map<string, double> initMap(const char* filename)
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
    *//*
    config.insert(make_pair("homework",0.1));
    config.insert(make_pair("quiz",0.15));
    config.insert(make_pair("lab",0.15));
    config.insert(make_pair("project",0.15));
    config.insert(make_pair("test",0.2));
    config.insert(make_pair("exam",0.25));
    *//*
    config[std::make_tuple(1, "homework")] = 0.1;
    config[std::make_tuple(2, "quiz")] = 0.15;
    config[std::make_tuple(3, "lab")] = 0.15;
    config[std::make_tuple(4, "project")] = 0.15;
    config[std::make_tuple(5, "test")] = 0.2;
    config[std::make_tuple(6, "exam")] = 0.25;
    */

    ifstream file (filename);
    // check file is opened
    if (!file) {
        cerr << "Error: file unable to be opened.\n";
        exit(2);
    }

    // load from file
    string category;
    double weight;
    while (file >> category >> weight)
        //config[category] = weight;
        config.insert(make_pair(category,weight));

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


void action(map<string, double>& config, const char* filename, bool& finished)
{
    if (DEBUG)
        printf("Entering action...\n");

    printMap(config);

    cout << "Would you like to:\n1. Add an assignment/weight\n2. Edit an assignment/weight\n3. "
         << "Calculate final grade\n4. Enter target grade\n5. Mark you are finished\n\nEnter your "
         << "choice: ";
    int choice;
    cin >> choice;

    switch(choice) {
        case 1:
        // add item/weight
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
        // edit item/weight
        {
            printMap(config);
            string item_edit;
            cout << "\nWhat item would you like to edit: ";
            cin >> item_edit;

            editMapItem(config, item_edit, filename);
            break;
        }

        case 3:
            // calculate final grade
            printf("Your final grade is %.2f\n", calcFinalGrade(config));
            break;

        case 4:
        // enter target grade -- this is for exam
        {
            double target_grade;
            cout << "What final grade would you like to receive: ";
            cin >> target_grade;
            
            printf("You need a %.2f on the exam to get a %.2f\n",finalExamCalc(config,target_grade),target_grade);
            break;
        }
        
        case 5:
            // mark as finished
            finished = true;
            break;

        // should not be hit if the user can read
        default:
            cerr << "Error: invalid operation choice";
            exit(3);
    }

    if (DEBUG)
        printf("Exiting action...\n");
}


void addMapItem(map<string, double>& config, const string& item, const double& weight, const char* filename)
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


void editMapItem(map<string, double>& config, const string& item, const char* filename)
{
    // the original approach would read from the file until it reached the target assignment,
    // then update the float that came after. Since I could not figure out how to do that, I 
    // instead decided to just remove and remake the file. Not the best solution, but it works
    
    if (DEBUG)
        printf("Entering editMapItem...\n");

    double weight;
    cout << "Enter the new value you want for " << item << ": ";
    cin >> weight;

    config[item] = weight;

    string command = "rm " + string(filename);
    int result = system(command.c_str());
    // check result
    if (result == -1)
        cerr << "Error executing command" << endl;

    ofstream file (filename);
    // check file is opened
    if (!file) {
        cerr << "Error: file unable to be opened.\n";
        exit(2);
    }

    for (const auto& entry : config)
        file << entry.first << " " << entry.second << "\n";
    
    file.close();

    if (DEBUG)
        printf("Exiting editMapItem...\n");
}


double calcFinalGrade(const map<string, double>& config)
{
    if (DEBUG)
        printf("Entering calcFinalGrade...\n");
    
    int* num_assignments = new int [config.size()];
    // array control for ^
    int index = 0;
    
    // get the number of assignments for each assignment
    int num_input;
    for (const auto& entry : config) {
        cout << "Enter the number of " << entry.first << " assignments: ";
        cin >> num_input;

        num_assignments[index] = num_input;
        index++;
    }
    // check all are accounted for
    if (index+1 != config.size())
        cerr << "Error: not all assignments accounted for\n";

    // calculations here
    double global_sum = 0, grade_sum = 0, grade;
    // inner for loop control var
    int j = 0;

    // for each assignment
    for (const auto& entry : config) {
        cout << "\n\nCurrent assignment category: " << entry.first << "\n";
        // for each assignment number (e.g. homework 1, homework 2, etc.)
        for(int i=0; i<num_assignments[j]; i++) {
            // get the grade for that assignment
            cout << toupper(entry.first[0]) << i+1 << ": ";
            cin >> grade;

            // add sum
            grade_sum += grade;
        }

        global_sum += (grade_sum/num_assignments[j]) * entry.second;

        // increment for next assignment number
        j++;
    }

    if (DEBUG)
        printf("Exiting calcFinalGrade...\n");
    
    return global_sum;
}


double finalExamCalc(const map<string, double>& config, const double& target_grade)
{
    if (DEBUG)
        printf("Entering finalExamCalc...\n");

    string final_assignment;
    cout << "What is the final assignment (e.g. exam, project, etc.): ";
    cin >> final_assignment;
    
    double final_weight;
    cout << "What is the weighted percentage of the final (enter XY%): ";
    cin >> final_weight;
    // input check
    while (final_weight >= 100) {
        cout << "Error: final assignment weight cannot exceed 100%. Re-enter: ";
        cin >> final_weight;
    }
    
    // define the weight of the current grade
    double current_grade_weight = 1-final_weight;

    if (DEBUG)
        printf("Exiting finalExamCalc...\n");

    return (target_grade - calcFinalGrade(config) * current_grade_weight)/final_weight;
}