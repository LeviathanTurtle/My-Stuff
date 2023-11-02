/* Name:
   Date:
   Class:
   Location:

   Implementation file for the derived class autoType:  
   Includes set, get, and simple mutator methods.
*/

#include"hybridType.h"
#include<iomanip>

hybridType::hybridType()
{
   chargeLevel = 0;
   chargeEff = 0;
}

hybridType::hybridType(double od_in, double fc_in, double fl_in, double mpg_in, double cl_in, double ce_in)
{
   autoType(od_in, fc_in, fl_in, mpg_in);

   if (cl_in < 0)
      chargeLevel = 0;
   else
      chargeLevel = cl_in;

   if(ce_in < 0)
      chargeEff = 0;
   else
      chargeEff = ce_in;   
}

void hybridType::setChargeLevel(double chargeLevel_in)
{
   if (chargeLevel_in > 0.0 && chargeLevel <= 100.0)
      chargeLevel = chargeLevel_in;
   else
      cerr << "Charge level must be > 0% and <= 100%" << endl;
}

double hybridType::getChargeLevel(void)const
{
   return chargeLevel;
}

void hybridType::setChargeEfficiency(double chargeEff_in)
{
   if (chargeEff_in > 1.0)
      chargeEff = chargeEff_in;
   else
      cerr << "Charge efficiency must be > 1.0" << endl;
}

double hybridType::getChargeEfficiency()const
{
   return chargeEff;
}

string hybridType::getAutoSpecs() const
{
   ostringstream hout;
   hout << fixed << showpoint << setprecision(2);
   hout << "Charge Level: " << chargeLevel << "%";
   hout << ", Charge Efficiency: " << chargeEff;

   autoType::getAutoSpecs();
   return hout.str();
}

void hybridType::drive(double hdistance)
{
   if(hdistance < 0)
   {
      cerr << "Cannot drive negative miles" << endl;
      return;
   }

   double hmaxDistance = fuelLevel*mpg + chargeLevel*chargeEff;
   if(hdistance <= hmaxDistance)
   {
      if(chargeLevel >= 20)
      {
         chargeLevel -= hdistance/chargeEff;
         hmaxDistance -= hdistance;
         odometer += hdistance;
         if(hmaxDistance != 0)
         {
            autoType::drive(hmaxDistance);
            while(hmaxDistance > 0)
            {
               chargeLevel++;
               hmaxDistance--;
            }
            if(fuelLevel <= 0)
               cerr << "Out of gas after: " << hmaxDistance << " miles" << endl;
         }
      }
   }
}






