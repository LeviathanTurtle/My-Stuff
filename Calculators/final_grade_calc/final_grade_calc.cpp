/* FINAL GRADE CALCULATOR
 * William Wadsworth
 * Created: at some point
 * 
 * [DESCRIPTION]:
 * This program calculates the final grade in a class. Assignment names and their weighted
 * percentages are stored in an external file named config by default, but specified at runtime.
 * Default assignments are homework, quizzes, tests, and labs. The user has the option to add
 * additional assignments with their weights or change an existing assignments weighted percentage,
 * which will all be saved in the same config file.
 * TODO: maybe a table or some kind of optional structured output, init config file if not found
 * 
 * [USAGE]:
 * To compile:
 *     g++ final_grade_calc.cpp -Wall -o <exe name>
 * To run:
 *     ./<exe name> [-d] <config file name>
 * where:
 * [-d] - enable debug output
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
#include <iomanip>      // spacing in output
#include <fstream>      // file I/O
#include <map>          // hold config options
#include <vector>       // only used in map sorting
#include <algorithm>    // sort()
#include <cstring>      // strcmp
//#include <utility>      // std::move
using namespace std;


bool DEBUG = false;


// function to initialize a map from the external file
map<string, double> initMap(const char*);
// function to print the current map
void printMap(const map<string, double>&);
// function that takes input from the user and does what the user specifies
void action(map<string, double>&, const char*, bool&);
// function to add an assignment to the current map
void addMapItem(map<string, double>&, const string&, const double&, const char*);
// function to edit a name of weight of an assignment
void editMapItem(map<string, double>&, const string&, const char*);
// function to output the map to a file
void dumpMap(map<string, double>&, const char*);
// function to calculate the final grade in a class
double calcFinalGrade(const map<string, double>&);
// function to calculate the grade you would need on a final to get a certain grade overall
double finalExamCalc(const map<string, double>&, const string&, const double&, const double&, double&);
// helper function to sort the map items
void mapSort(map<string, double>&);


int main(int argc, char* argv[])
{
    // check CLI args
    if(argc < 2 || argc > 3) {
        cerr << "Error: invalid arguments. Usage: ./<exe name> [-d] <config file name> \n";
        exit(1);
    }

    // call loadFile func
    // BASED ON CLI ARGS
    //if (argv[1] == "-d") {
    if (!strcmp(argv[1],"-d")) {
        // debug
        DEBUG = true;

        // initialize map
        map<string, double> config = initMap(argv[2]);

        // prompt user for action
        // bool var for repeating inputs
        bool finished = false;
        // repeat until the user specified they are finished
        while (!finished)
            action(config, argv[2], finished);
    }
    else {
        // not debug

        // initialize map
        map<string, double> config = initMap(argv[1]);

        // prompt user for action
        // bool var for repeating inputs
        bool finished = false;
        // repeat until the user specified they are finished
        while (!finished)
            action(config, argv[1], finished);
    }

    return 0;
}


/* function to initialize a map from the external file
 * pre-condition: expects a character pointer (string) filename as input, which should not be null
 *                or empty. This will be used to populate the key-value pair map
 * 
 * post-condition: populates a key-value pair map (string, double) named `config` which is
 *                 returned. It is populated from input via an external file with the format
 *                 'string double\nstring double\n...'. The map is then sorted from least to
 *                 greatest based on the 'double' value
*/
map<string, double> initMap(const char* filename)
{
    if (DEBUG)
        printf("Entering initMap...\n");

    // declare map object
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
    config[make_tuple(1, "homework")] = 0.1;
    config[make_tuple(2, "quiz")] = 0.15;
    config[make_tuple(3, "lab")] = 0.15;
    config[make_tuple(4, "project")] = 0.15;
    config[make_tuple(5, "test")] = 0.2;
    config[make_tuple(6, "exam")] = 0.25;
    */

    // set up file 
    ifstream file (filename);
    // check file is opened
    if (!file) {
        cerr << "Error: file unable to be opened.\n";
        exit(2);
    }

    // load from file
    string category;
    double weight;
    // while there are file contents, add them to the map
    while (file >> category >> weight)
        //config[category] = weight;
        config.insert(make_pair(category,weight));

    file.close();
    // sort the map
    mapSort(config);

    if (DEBUG) {
        printf("Loaded map:\n");
        printMap(config);
        printf("\nExiting initMap...\n");
    }

    return config;
}


/* function to print the current map
 * pre-condition: expects a reference to a key-value pair map `config` of strings to doubles as
 *                input
 * 
 * post-condition: prints each key-value pair from the input map `config`. The output format is
 *                 "<key>: <value>" where <key> is a string and <value> is a double, with each
 *                 key-value pair printed on a new line
*/
void printMap(const map<string, double>& config)
{
    if (DEBUG)
        printf("Entering printMap...\n");

    double total = 0;
    size_t max_item_length = 0;

    // find length of the longest item
    for (const auto& entry : config)
        max_item_length = max(max_item_length, entry.first.length());

    for (const auto& entry : config) {
        //cout << entry.first << ": " << entry.second << "\n";
        total += entry.second*100;
        //printf("%s: %.2f ( %d%% )\n", entry.first.c_str(), entry.second, int(entry.second*100));
        cout << setw(max_item_length+2) << left << entry.first+":" << setw(5) << entry.second
             << " ( " << int(entry.second*100) << " % )\n";
    }
    cout << "Total: " << setw(max_item_length+6) << right << int(total) << "%" << endl;

    if (DEBUG)
        printf("Exiting printMap...\n");
}


/* function that takes input from the user and does what the user specifies
 * pre-condition: expects a reference to a key-value pair map `config` of strings to doubles as
 *                input. It also expects a reference to a character pointer (string) filename which
 *                should not be null or empty and an initialized boolean value
 * 
 * post-condition: calls printMap, takes a user's input to indicate which function to call.
 *                 Depending on the function call, 'config' may be updated or re-created. If an
 *                 invalid option is selected, the program exits
*/
void action(map<string, double>& config, const char* filename, bool& finished)
{
    if (DEBUG)
        printf("Entering action...\n");

    // print the current config for the user
    cout << "------------------------------" << endl;
    printMap(config);

    // check that all weights = 1 (100%)
    double weight_check = 0;
    // sum the weights of each assignment
    for (const auto& entry : config)
        weight_check += entry.second;
    // if the sum of the weights do not equal 1 (100%), output warning
    if(weight_check != 1)
        cerr << "Warning: weights do not sum to 1 (100%)!\n";

    cout << "\nWould you like to:\n1. Add an assignment/weight\n2. Edit an assignment/weight\n3. "
         << "Calculate final grade\n4. Enter target grade\n5. Exit program\n\nEnter your choice: ";
    int choice;
    cin >> choice;
    // TODO: ADD REMOVE ITEM OPTION

    double final_grade = -1.0;

    switch(choice) {
        // add item/weight
        case 1:
        {
            string item;
            cout << "What is the assignment name? Enter as one word: ";
            cin >> item;
            double weight;
            cout << "What is the assignment's weight? Enter as a decimal: ";
            cin >> weight;

            addMapItem(config, item, weight, filename);
            break;
        }
        // edit item/weight
        case 2:
        {
            printMap(config);
            string item_edit;
            cout << "\nWhat item would you like to edit: ";
            cin >> item_edit;

            editMapItem(config, item_edit, filename);
            break;
        }
        // calculate final grade
        case 3:
            final_grade = calcFinalGrade(config);
            printf("Your final grade is %.2f%%\n", final_grade);
            break;
        // enter target grade -- this is for final exam
        case 4:
        {
            string final_assignment;
            cout << "What is the final assignment name: ";
            cin >> final_assignment;
            double final_weight;

            // ensure that final is in the map
            bool found = false;
            for (const auto& entry : config)
                if (entry.first == final_assignment) {
                    char confirmation;
                    cout << "Is " << entry.first << " the final assignment? [Y/n]: ";
                    cin >> confirmation;

                    if (confirmation == 'Y') {
                        found = true;
                        final_weight = entry.second;
                    }
                }
                    
            // if it is not in the map, add it
            if (!found) {              
                cout << "What is the final assigment weight? Enter as a decimal: ";
                cin >> final_weight;

                addMapItem(config,final_assignment,final_weight,filename);
            }
            
            // get final weight
            double target_grade;
            cout << "What final grade would you like to receive: ";
            cin >> target_grade;
            
            printf("With a current grade of %.2f, you need a %.2f on the exam to get a %.2f\n",final_grade,finalExamCalc(config,final_assignment,final_weight,target_grade,final_grade),target_grade);
            break;
        }
        // mark as finished
        case 5:
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


/* function to add an assignment to the current map
 * pre-condition: expects a reference to a key-value pair map `config` of strings to doubles as
 *                input, a constant reference to a string and double, and a reference to a
 *                character pointer (string) filename which should not be null or empty
 * 
 * post-condition: adds the string 'item' and double 'weight' pair to the map 'config', then dumps
 *                 the contents of the map to an external file via dumpMap 
*/
void addMapItem(map<string, double>& config, const string& item, const double& weight, const char* filename)
{
    if (DEBUG)
        printf("\nEntering addMapItem...\n");

    // add the item to the map
    config.insert(make_pair(item,weight));

    // output map to file
    dumpMap(config, filename);

    if (DEBUG) {
        printf("Updated map:\n");
        printMap(config);
        printf("\nExiting addMapItem...\n");
    }
}


/* function to output the map to a file
 * pre-condition: expects a reference to a key-value pair map `config` of strings to doubles as
 *                input and a reference to a character pointer (string) filename which should not
 *                be null or empty
 * 
 * post-condition: sorts the map for redundancy and outputs the contents of the map to an external
 *                 file with the format "string double\nstring double\n..."
*/
void dumpMap(map<string, double>& config, const char* filename)
{
    if (DEBUG)
        printf("Entering dumpMap...\n");
    
    // ensure map is sorted
    //mapSort(config);

    // remove the file
    string command = "rm " + string(filename);
    int result = system(command.c_str());
    // check result
    if (result == -1)
        cerr << "Error executing command" << endl;
    
    // set up file output
    ofstream file (filename);
    // check file is opened
    if (!file) {
        cerr << "Error: file unable to be opened.\n";
        exit(2);
    }

    for (const auto& entry: config)
        // output new assignment to file
        file << entry.first << " " << entry.second << endl;

    file.close();

    if (DEBUG)
        printf("Exiting dumpMap...\n");
}


/* function to edit a name of weight of an assignment
 * pre-condition: expects a reference to a key-value pair map `config` of strings to doubles as
 *                input, a constant reference to a string, and a reference to a character pointer
 *                (string) filename which should not be null or empty
 * 
 * post-condition: the original config file is deleted and re-made, assuming the weights sum to 1.
 *                 If not, the function is re-called to try again
*/
void editMapItem(map<string, double>& config, const string& item, const char* filename)
{
    // the original approach would read from the file until it reached the target assignment,
    // then update the float that came after. Since I could not figure out how to do that, I 
    // instead decided to just remove and remake the file. Not the best solution, but it works
    
    if (DEBUG)
        printf("Entering editMapItem...\n");

    // check that item is in map
    bool found = false;
    for (const auto& entry : config)
        if (entry.first == item)
            found = true;

    double weight;
    cout << "Enter the new value you want for " << item << ": ";
    cin >> weight;

    if (!found)
        addMapItem(config,item,weight,filename);
    else
        config[item] = weight;

    // check that all weights = 1 (100%)
    double weight_check = 0;
    // sum the weights of each assignment
    for (const auto& entry : config)
        weight_check += entry.second;
    // if the sum of the weights do not equal 1 (100%), output error, re-call function
    if(weight_check != 1) {
        cerr << "Warning: weights do not sum to 1 (100%)\n";
        //printMap(config);
        //editMapItem(config, item, filename);
    }

    // output map to file
    dumpMap(config, filename);

    if (DEBUG)
        printf("Exiting editMapItem...\n");
}


/* function to calculate the final grade in a class
 * pre-condition: expects a constant reference to a key-value pair map `config` of strings to
 *                doubles as input
 * 
 * post-condition: returns the final grade based on the number of assignments given. These values
 *                 are input by the user
*/
double calcFinalGrade(const map<string, double>& config)
{
    if (DEBUG)
        printf("Entering calcFinalGrade...\n");
    
    // var to keep track of number of assignments processed, will be used to check all are used
    int num_input, num_assignments_used = 0;
    // calculation vars
    double grade, global_sum = 0;
    
    // get the number of assignments for each assignment
    for (const auto& entry : config) {
        cout << "Enter the number of " << entry.first << " assignments: ";
        cin >> num_input;

        //cout << "\nCurrent assignment category: " << entry.first << "\n";
        double grade_sum = 0;

        // for each assignment number (e.g. homework 1, homework 2, etc.)
        for(int i=1; i<=num_input; i++) {
            // get the grade for that assignment
            cout << (char)toupper(entry.first[0]) << entry.first.substr(1) << " " << i << ": ";
            cin >> grade;

            // add to assignment sum
            grade_sum += grade;
            // todo : maybe scanf instead?
        }

        // add to total sum
        global_sum += (grade_sum/num_input) * entry.second;

        // increment for current processed item
        num_assignments_used++;
    }

    // check all are accounted for
    if (num_assignments_used != int(config.size()))
        cout << "Warning: not all assignments accounted for!\n";

    if (DEBUG)
        printf("Exiting calcFinalGrade...\n");
    
    return global_sum;
}


/* function to calculate the grade you would need on a final to get a certain grade overall
 * pre-condition: expects a constant reference to a key-value pair map `config` of strings to
 *                doubles as input and a constant reference to a double
 * 
 * post-condition: the function returns the grade you would need to score on the final to get
 *                 the grade the user specifies
*/
double finalExamCalc(const map<string, double>& config, const string& final_assignment, const double& final_weight, const double& target_grade, double& current_grade)
{
    if (DEBUG)
        printf("Entering finalExamCalc...\n");

    // define the weight of the current grade
    double current_grade_weight = 1-final_weight;

    if (DEBUG)
        printf("Exiting finalExamCalc...\n");

    if (current_grade == -1)
        current_grade = calcFinalGrade(config);

    return (target_grade - current_grade * current_grade_weight)/final_weight;
}


/* helper function to sort the map items
 * pre-condition: expects a reference to a key-value pair map `config` of strings to doubles as
 *                input
 * 
 * post-condition: the map is cleared, then re-created sorted based from least to greatest based
 *                 on the 'double' value 
*/
// NOTE: this does not work completely
void mapSort(map<string, double>& config)
{
    if (DEBUG)
        printf("Entering mapSort...\n");

    // create a vector copy because maps cannot be sorted
    vector<pair<string, double> > vec(config.begin(), config.end());

    // sort vector based on double (lambda func)
    sort(vec.begin(), vec.end(),
        [](const pair<string, double>& a, const pair<string, double>& b) {
             return a.second < b.second;
         });

    // clear original map
    config.clear();

    // put sorted entries back into map
    //for (const auto& entry : vec)
    //    config.insert(entry);

    // move sorted entries back into map (efficient and preserves order)
    //move(vec.begin(), vec.end(), config.begin());

    copy(vec.begin(), vec.end(), inserter(config, config.end()));

    if (DEBUG)
        printf("Exiting mapSort...\n");
}

