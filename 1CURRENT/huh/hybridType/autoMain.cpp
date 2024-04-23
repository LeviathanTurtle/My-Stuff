/*
   Name: 
   Date: 
   Class:
   Location: 

   To compile, use command:
      g++ -Wall autoMain.cpp autoType.cpp hybridType.cpp -o lab6Test		
   To execute, use command
      ./lab6Test
*/

#include"hybridType.h"
#include"autoType.h"
#include<iomanip>

int main()
{  

   //test parameterized constructor
   autoType F250(100.5, 20, 16.25, 19.7);

   cout << "******FORD F250******" << endl;
   cout << F250.getAutoSpecs() << endl << endl;

   //test addFuel
   cout << "Adding 10 gallons." << endl;
   F250.addFuel(10);
   cout << F250.getAutoSpecs() << endl << endl;

   //test drive
   cout << "Requesting to drive 400 miles." << endl;
   F250.drive(400);
   cout << F250.getAutoSpecs() << endl << endl;

   //test default constructor
   hybridType fusion;
   cout << "******Ford Fusion******" << endl;
   cout << fusion.getAutoSpecs() << endl << endl;

   //test setAutoSpecs
   cout << "Setting auto specs to 1234, 2.25, and 42.7" << endl;
   fusion.setAutoSpecs(1234, 2.25, 42.7);
   cout << fusion.getAutoSpecs() << endl << endl;

   //test set methods of hybridType
   cout << "Setting charge level and Efficiency to 15.5 and 10.7" << endl;
   fusion.setChargeLevel(15.5);
   fusion.setChargeEfficiency(10.7);

   //test get methods of hybridType
   cout << "Charge Level: " << fusion.getChargeLevel() << endl;
   cout << "Charge Efficiency: " << fusion.getChargeEfficiency() << endl << endl;

   //test addFuel
   cout << "Adding 15 gallons" << endl;
   fusion.addFuel(15);
   cout << fusion.getAutoSpecs() << endl << endl;

   //test drive
   cout << "Request to drive 200 miles." << endl;
   fusion.drive(200);
   cout << fusion.getAutoSpecs() << endl;

   return 0;
}
