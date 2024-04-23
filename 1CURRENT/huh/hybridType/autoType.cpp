/* Name: 
   Date:
   Class:
   Location:

   Implementation file for the class autoType.
   Includes set, get, and simple mutator methods.
*/

#include"autoType.h"
#include<iomanip>

autoType::autoType(double od_in, double fc_in, double fl_in, double mpg_in)
{
   //Set to defaults then adjust if valid.
   odometer=0;
   fuelCap=20;
   fuelLevel=0;
   mpg=1.0;
   if(od_in > 0)
      odometer=od_in;
   if(fc_in > 0)
      fuelCap=fc_in;
   if(fl_in > 0)
      fuelLevel=fl_in;
   if(mpg_in > 1.0)
      mpg=mpg_in;
}


void autoType::setAutoSpecs(double od_in, double fl_in, double mpg_in)
{
   //Assume parameters are acceptable
   bool isValid = true;
   
   //Input Validation
   if (od_in < odometer)
   {
      cerr << "Cannot set a vehicle's odometer to a value less than current" 
           << endl;
      isValid = false;
   }
   if (mpg_in < 1)
   {
      cerr << "Fuel Efficiency must be greater than 1" << endl;
      isValid = false;
   }
   if (fl_in > fuelCap)
   {
      cerr << "Fuel Level cannot exceed Fuel Capacity" << endl;
      isValid = false;
   }

   //SetSpecs
   if (isValid)
   {
      odometer = od_in;
      fuelLevel = fl_in;
      mpg = mpg_in;
   } else {
      cerr << "No updates were made to auto specs" << endl;
   }
}


string autoType::getAutoSpecs(void)const
{
   ostringstream sout;
   sout << fixed << showpoint << setprecision(2);
   sout << "Odometer = " << odometer;
   sout << ", Fuel Capacity = " << fuelCap;  
   sout << ", Fuel Level = " << fuelLevel;
   sout << ", Efficiency = " << mpg;

   return sout.str();
}
      
void autoType::drive(double distance)
{
   //cannot drive a negative distance
   if (distance < 0)
   {
      cerr << "Cannot drive negative miles." << endl;
      return;
   }
 
   //Determine the max dist you can travel 
   //based on the current fuelLevel.
   double maxDistance = fuelLevel*mpg; 

   //Compare the requested distance to maxDistance
   //to see if that is possible and act accordingly.
   if(distance <= maxDistance) 
   {
      odometer += distance;
      fuelLevel -= distance/mpg;
   } 
   else 
   {
      cerr << "Out of gas after " << maxDistance << " miles." << endl;
      odometer += maxDistance;
      fuelLevel = 0;
   }
}

void autoType::addFuel(double fill)
{
   //Cannot have a -fill 
   if (fill<0)
   {
      cerr << "Cannot fill less than zero gas" << endl;
      return; 
   }
 
   cout << "Request to fill " << fill << " gallons of gas." << endl;

   //Detirmine the max fuel you can add
   //based on the current fuelLevel
   double maxFill = fuelCap - fuelLevel;

   //Compare the requested fill to maxFill
   //to see if that is possible and act accordingly
   if (fill <= maxFill)
      fuelLevel += fill;
   else
   {
      double tankFill = fuelCap - fuelLevel;
      fuelLevel = fuelCap;
      cout << "Tank filled after " << tankFill << " gallons." << endl;
   }
}
