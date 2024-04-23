#ifndef COUNTER_TYPE
#define COUNTER_TYPE

class counterType
{
   private:
      int counter;
   public:
      void initializeCounter();
      void setCounter(int c = 0);
      int getCounter() const;
      void incrementCounter();
      void decrementCounter();
      void displayCounter() const;
      counterType(int t);
      counterType();
};

#endif
