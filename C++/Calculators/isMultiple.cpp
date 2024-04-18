/* IS ONE INTEGER A MULTIPLE OF THE OTHER
 *
 * William Wadsworth
 * Created: 10.11.2020
 * Doctored: 10.12.2023
 * CSC1710
 * ~/csc1710/chap5/num5.cpp
 * 
 * 
 * [SUMMARY]:
 * This program takes two integers X and Y and determines if X is a multiple of
 * Y. The integers are passed as CLI arguments using argc and argv. There
 * should only be 3 arguments: the exe and the two integers. If X is a multiple
 * of Y, the program will calculate and output each factor until it reaches X. 
 * 
 * 
 * [USE]:
 * To compile: g++ isMultiple.cpp -Wall -o isMultiple
 * To run: ./isMultiple <X> <Y>
 * 
 * where <X> and <Y> are the integers you want to use.
 * 
 * Restrictions: X and Y must be greater than 0 and rounded to the nearest
 *               whole
 * 
 * 
 * [EXAMPLE RUN]:
 * 
 * ./isMultiple 50 10
 * 
 * will output the following:
 * 10
 * 20
 * 30
 * 40
 * 50
 * 
 * 50 has 5 multiples of 10
*/

#include <iostream>
using namespace std;


int main(int argc, char* argv[])
{
    // check correct args are given
    if(argc != 3) {
        cerr << "error: invalid number of arguments. " << argc << " provided.\n";
        exit(1);
    }

    // sum/count variables for calculation
    int sum=0, count=0;

    // convert char* to integer
    int x = atoi(argv[1]);
    int y = atoi(argv[2]);

    // check if x is divisible by y
    if( x % y == 0) {
        // calculation of multiples
        while(sum < x) {
            sum += y;
            cout << sum << endl;
            count++;
        }

        // output results
        cout << endl << x << " has " << count << " multiples of " << y << endl;
    }
    else
        cout << x << " has no multiples of " << y << endl;

    
    return 0;
}

