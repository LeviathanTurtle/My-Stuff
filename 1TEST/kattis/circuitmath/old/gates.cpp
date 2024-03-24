#include <iostream>
#include <vector>
#include <stack>
using namespace std;

struct component
{
    char name, logic;
};

bool inpCheck();

int main()
{
    int gates;
    cin >> gates;
    char num, letter;
    
    //vector<char> logic[4];
    char logic[4];
    
    component part[4];
    //char part[4];
    stack<char> stack;
    
    // get T/F
    for(int i=0; i<4/*num*/; i++) {
        cin >> part[i].logic;
        cout << part[i].logic << " ";
    }
    // get A, B, C, ...
    for(int i=0; i<8; i++) {
	if(inpCheck())
	    cin >> part[i].name;
    }
    // get gate names/operators
    char op; int j=0;
    component atemp, btemp;
    while(cin.peek()!='\n')
    {
        if(inpCheck())
	{
	    cin >> letter;
	    stack.push(letter);
	}
	// *
	if(cin.peek()==42)
	{
	    cin >> op;
	    btemp.name=stack.pop(); btemp.logic=part[j];
	    atemp.name=stack.pop(); atemp.logic=part[j+1];
	    if(atemp.logic=='T' && btemp.logic=='T')
		stack.push('T');
	    else
		stack.push('F');
	}
	// +
	if(cin.peek()==43)
	{
	    cin >> op;
	    atemp=stack.pop(); btemp=stack.pop();
	    if(atemp.logic=='F' && btemp.logic=='F')
		stack.push('F','F');
	    else
		stack.push('T','T');
	}

	// -
	if(cin.peek()==45)
	{
	    cin >> op;
	    atemp=stack.pop();
	    if(atemp.logic=='T')
		stack.push('F','F');
	    else
		stack.push('T','T');
	}
    }
    
    return 0;
}

bool inpCheck()
{
    char t;
    if(cin.peek()>=65 && cin.peek()<=90)
	return true;
}
