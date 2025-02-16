#include <iostream>
#include <stack>
using namespace std;

int main()
{
    int i, numvars, logic[4];
    char eof;
    cin >> numvars >> eof;

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
        //cin >> logTemp;
    }
    cin >> eof;

    // LOGIC
    int name, op1, op2, symb;
    i=0;
    while(!cin.eof())
    //while(cin.peek()!='\n')
    {
        // A,B,C,...
        if(cin.peek()>=65 && cin.peek()<=90)
        {
            cin >> name;
            stack.push(logic[i]);
            i++;
            cout << "logread\n";
        }
        // *
        if(cin.peek()==42)
        {
            cin >> symb;
            op2=stack.top();
            stack.pop();
            op1=stack.top();
            stack.pop();
            stack.push(op1&op2);
            cout << "opdone\n";
        }
        // +
        if(cin.peek()==43)
        {
            cin >> symb;
            op2=stack.top();
            stack.pop();
            op1=stack.top();
            stack.pop();
            stack.push(op1|op2);
            cout << "opdone\n";
        }
        // -
        if(cin.peek()==45)
        {
            cin >> symb;
            op1=stack.top();
            stack.pop();
            stack.push(!op1);
            cout << "opdone\n";
        }
    }
    if(stack.top()==0)
        cout << 'F';
    else
        cout << 'T';

    return 0;
}
