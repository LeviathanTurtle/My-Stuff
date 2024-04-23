/* Name:
   Date:
   Class:
   Location:
*/
#include<iostream>
using namespace std;

#ifndef AUTO_TYPE
#define AUTO_TYPE

class autoType
{
   protected:
      double fuelCap;
      double fuelLevel;
      double mpg;
      double odometer;
   public:

      /* Constructor for the class with parameters
       * Sets the odometer, fuelCap, fuelLevel and mpg. 
       * If no paramters are given it uses the default values.
       * Postcondition - odometer=od_in, fuelCap = fc_in
       *                 fuelLevel=fl_in, mpg=mpg_in
       * NOTE: with this constructor, a default constructor is
       * NOT needed.
       */
      autoType(double od_in=0, double fc_in=20, double fl_in=0, double mpg_in=1);
      
      /* Function to set the auto specs - odometer, fuelLevel, and mpg.
       * Postcondition - odometer=od_in, fuelLevel=fl_in
       *                 mpg=mpg_in
       */
      void setAutoSpecs(double od_in, double fl_in, double mpg_in);

      /* Function to build a string containing the odometer, fuelLevel,
       * and mpg.  The values will be rounded off to two decimal places.
       * Postcondition - the string is returned.
       */
      string getAutoSpecs(void)const;

      /* Function to drive the car the distance given in a parameter.
       * The function will update the odometer and fuel amounts and
       * monitor if you are trying to drive more than you have fuel.
       * If you do not have enough fuel to drive the distance, the 
       * odometer and fuel will be updated based on how far you actually
       * can drive.
       */ 
      void drive(double distance);
      
      /* The function will update the fuel amounts and monitor if you are 
       * trying to add more fuel than you have capcity for. If you add too 
       * much fuel, your fuelLevel will equal the fuelCap, and the function
       * will tell you how much it filled of what you input.
       */
      void addFuel(double fill);
     
};

#endif
