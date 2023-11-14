/* LOGIC CIRCUIT MATH
 * William Wadsworth
 *   Professor Shore: suggested stack approach, approach to put respective
 *                    value onto stack
 * Created: 
 * CSC - KATTIS
 *
 *
 * [DESCRIPTION]:
 * This program performs circuit math taken from an input file using I/O
 * redirection. The program outputs the outcome of the circuit.
 * 
 * 
 * [COMPILE/RUN]:
 * To compile:
 *     g++ gates.cpp -Wall -o gates
 * 
 * To run:
 *     ./gates < <data file>
 * 
 * 
 * [DATA FILE STRUCTURE]:
 * <N>
 * <gate 1 logic value> <gate 2 logic value> ... <gate N logic value>
 * <N letter values with operators>
 * 
 * where N is the number of gates
 * Note: the order of gates must be alphabetical for this program to make sense
 *       based on the gate names
 * 
 * 
 * [DATA FILE EXAMPLE]:
 * 4
 * T F T F
 * A B * C D + - +
 *
 * 
 * [EXIT/TERMINATING CODES]:
 * 0 - program successfully completed a full execution
*/

#include <iostream>
#include <stack>
using namespace std;

int main()
{
    // array for gates
    // set to 26 because there are 26 alphabet characters
    int gates[26];

    // get number of gates
    int numGates;
    cin >> numGates;

    // define stack, set to int:
    // 0 - false
    // 1 - true
    stack<int> stack;

    // variable to read in logic T/F
    char logicTemp;
    // read in logic, repeat for the number of gates
    for(int i=0; i<numGates; i++) {
        cin >> logicTemp;
        if(logicTemp=='T')
            gates[i]=1;
        else
            gates[i]=0; 
    }

    // LOGIC
    // temporary variables for operations
    int op1, op2;
    // variable for character in input stream
    char x;

    // repeat while there is input
    while(cin >> x) {
        switch(x) {
            case '*':
                op2=stack.top();
                stack.pop();
                op1=stack.top();
                stack.pop();
                // logical and 
                stack.push(op1&op2);
                break;
            case '+':
                op2=stack.top();
                stack.pop();
                op1=stack.top();
                stack.pop();
                // logical or
                stack.push(op1|op2);
                break;
            case '-':
                op1=stack.top();
                stack.pop();
                // logical not
                stack.push(!op1);
                break;
            default:
                // put the respective letter onto the stack

                // example: the current gate x = D
                //     ascii value of D = 68, A = 65
                //
                //     gates[68-65] = gates[3]
                //     gates[3] = 4th letter in alphabet
                stack.push(gates[x-'A']);
                break;
        }
    }

    // stack is complete, read the top value for the final outcome
    if(stack.top()==0)
        cout << "F\n";
    else
        cout << "T\n";

    return 0;
}
