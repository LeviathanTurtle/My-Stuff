/* 
   Name: William Wadsworth
   Date: 2.9.21
   Class: CSC1720
   Location: ~/csc1720/lab5/autoType.cpp

   This file contains the implementation of both classes
*/

#include"autoType.h"
#include<iomanip>

autoType::autoType(double od_in, double fl_in, double mpg_in, double fuelCap_in)
{
   fuelCap = fuelCap_in;
   setAutoSpecs(od_in, fl_in, mpg_in);
}

void autoType::setAutoSpecs(double od_in, double fl_in, double mpg_in)
{
   odometer=fuelLevel=mpg=0;

   if (od_in >= 0 && fl_in >= 0 && fl_in <= fuelCap && mpg_in >= 0)
   {
      odometer = od_in;
      fuelLevel = fl_in;
      mpg = mpg_in;
   }
   else
   {
      cerr << "Invalid value(s)" << endl;
      exit(1);
   }
}

string autoType::getAutoSpecs(void)const
{
   ostringstream sout;

   sout << fixed << showpoint << setprecision(2);
   sout << "Miles = " << odometer;
   sout << ", Fuel Level = " << fuelLevel;
   sout << ", Efficiency = " << mpg;
   sout << ", Fuel Capacity = " << fuelCap;

   return sout.str();
}
      
void autoType::drive(double distance)
{
   //Determine the max dist you can travel 
   //based on the current fuelLevel.
   //Compare the requested distance to maxDistance
   //to see if that is possible and act accordingly.
   double maxDistance = fuelLevel*mpg; 
   if(distance <= maxDistance) {
      odometer += distance;
      fuelLevel -= distance/mpg;
   } else {
      cerr << "Out of gas after " << maxDistance << " miles." << endl;
      odometer += maxDistance;
      fuelLevel = 0;
   }
}

void autoType::addFuel(double amount)
{
   fuelLevel += amount;
   if (fuelLevel > fuelCap)
      cerr << "Your fuel overflowed" << endl;
}

void hybridType::setChargeLevel(double charge_in)
{
   if (charge_in >= 0)
      chargeLevel = charge_in;
   else
      chargeLevel = 0;
}

double hybridType::getChargeLevel() const
{
   cout << fixed << showpoint << setprecision(2);
   // 
   return chargeLevel;
}

void hybridType::setChargeEfficiency(double chargeEff_in)
{
   if (chargeEff_in >= 0)
      chargeEff = chargeEff_in;
   else
      chargeEff = 0;
}

double hybridType::getChargeEfficiency() const
{
   cout << fixed << showpoint << setprecision(2);
   // 
   return chargeEff;
}



