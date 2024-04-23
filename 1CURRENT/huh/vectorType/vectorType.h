/*

*/

#ifndef VECTOR_TYPE
#define VECTOR_TYPE

class vectorType
{
   private:
      double x;
      double y;
      double z;
   public:
      vectorType(double, double, double);
      vectorType();
      void setComps(double, double, double);
      double getX() const;
      double getY() const;
      double getZ() const;
      void printVector() const;
      double calcMagnitude(double, double, double);
      void scalarMultiply(double, double, double, double);
};

#endif
