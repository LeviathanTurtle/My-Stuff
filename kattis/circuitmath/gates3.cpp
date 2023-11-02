#include <iostream>
#include <stack>
using namespace std;

int main()
{
    int i=0, numvars, logic[26];
    cin >> numvars;

    stack<int> stack;

    // T/F
    char logTemp;
    for(i=0; i<numvars; i++)
    {
        cin >> logTemp;
        if(logTemp=='T')
            logic[i]=1;
        else
            logic[i]=0; 
    }

    // LOGIC
    int op1, op2;
    char x;
    i=0;
    while(cin >> x)
    {
        switch(x)
        {
            case '*':
                op2=stack.top();
                stack.pop();
                op1=stack.top();
                stack.pop();
                stack.push(op1&op2);
                break;
            case '+':
                op2=stack.top();
                stack.pop();
                op1=stack.top();
                stack.pop();
                stack.push(op1|op2);
                break;
            case '-':
                op1=stack.top();
                stack.pop();
                stack.push(!op1);
                break;
            default:
                  stack.push(logic[x-'A']);
                break;
        }
    }

    if(stack.top()==0)
        cout << "F\n";
    else
        cout << "T\n";

    return 0;
}
