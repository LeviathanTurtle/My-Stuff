/* 
   Author: William Wadsworth
   Date: 2.9.21
   Class: CSC1720
   Code location: ~/csc1720/lab5/autoMain.cpp

   About:
   This program will output the odometer, fuel level, fuel efficiency, and fuel
      capacity. 

   To compile:
      g++ -Wall autoMain.cpp -o  vehicleRangeCalculator

   To execute:
      ./vehicleRangeCalculator
*/

#include "autoType.h"
#include "autoType.cpp"
#include <iomanip>

int main()
{  
   autoType fordF250;
   autoType dodgeRam(201, 10.32, 19.1, 100);

   fordF250.setAutoSpecs(1234,16.25,25.7);
   cout << fordF250.getAutoSpecs() << endl;
   cout << "Requesting to drive 400 miles." << endl;
   fordF250.drive(400);
   cout << fordF250.getAutoSpecs() << endl;
   fordF250.addFuel(27); 
   cout << endl;


   cout << dodgeRam.getAutoSpecs() << endl;
   cout << "Requesting to drive 400 miles." << endl;
   dodgeRam.drive(400);
   dodgeRam.addFuel(50);
   cout << dodgeRam.getAutoSpecs() << endl;
   cout << endl;


   hybridType hybrid1;
   //hybridType hybrid2;

   cout << hybrid1.getAutoSpecs() << endl;
   cout << "Charge: " << hybrid1.getChargeLevel() << "%" << endl;
   cout << "Efficiency: " << hybrid1.getChargeEfficiency() << " kWh" << endl;
   cout << endl;

   hybrid1.setChargeLevel(200);
   hybrid1.setChargeEfficiency(150);

   cout << hybrid1.getAutoSpecs() << endl;
   cout << "Charge: " << hybrid1.getChargeLevel() << "%" << endl;
   cout << "Efficiency: " << hybrid1.getChargeEfficiency() << " kWh" << endl;

   return 0;
}
