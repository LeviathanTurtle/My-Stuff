/* Name: 
   Date:
   Class: 
   Location: 

   this is the header file for the derived autoType class: hybridType
   hybridType is a type of autoType, in which it has two more 
   private instance variable,
                 the amount of charge in its batteries --> chargeLevel
                 and the battery/charge efficiency --> chargeEff
*/
#include<iostream>
#include "autoType.h"
using namespace std;

#ifndef HYBRID_TYPE
#define HYBRID_TYPE

class hybridType: public autoType
{
   private:
      double chargeLevel;
      double chargeEff;
   public:
      /*
      hybridType - default constructor
      pre-condition: private members in class are defined
      post-condition: each private member in the classes are set to their input values
      */
      hybridType();

      /*
      hybridType - paramterized constructor for all 6 values
      pre-condition: private members in both classes are defined
      post-condition: each private member in the classes are set to their input values
      */
      hybridType(double od_in, double fc_in, double fl_in, double mpg_in, double cl_in, double ce_in);

      /*
      setChargeLevel - use to update the private instance variable
                       chargeLevel of a hybridType.  
                       Max change level is 100%.
      pre-conditions: parameter must be a double in percent form
      post-conditions: chargeLevel = chargeLevel_in
      */
      void setChargeLevel(double chargeLevel_in);

     /*
      getChargeLevel - use to "get" the value of chargeLevel 
                       when it is not directly available
                       ((when outside the class))
      post-conditions: returns chargeLevel in percent form
      */
      double getChargeLevel(void)const;

     /*
      setChargeEfficiency - use to update the private instance variable
                            chargeEff of a hybridType
      pre-conditions: parameter must be a double
      post-conditions: chargeEff = chargeEff_in
      */
      void setChargeEfficiency(double chargeEff_in);

     /*
      getChargeEfficiency - use to "get" the value of chargeEff
                            when it is not directly available 
                            ((outside the class))
      post-conditions: returns chargeEff
      */
      double getChargeEfficiency()const;
      /*
      getAutoSpecs - retrieves vehicle values from class
      pre-condition: values must be defined and initialized
      post-condition: returns data values
      */
      string getAutoSpecs() const;

      /*
      drive - adds miles to odometer and subtracts used fuel or charge
      pre-doncition: distance, charge, fuel must be defined and initialized
      post-condition: nothing is returned
      */
      void drive(double hdistance);








};

#endif
