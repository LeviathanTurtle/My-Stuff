#ifndef UnorderedArrayList_TYPE
#define UnorderedArrayList_TYPE
   
#include "arrayListType.h" 

class unorderedArrayListType: public arrayListType
{
   public:
      //Constructor
      unorderedArrayListType(int size = 100);

      /* insertAt Function ...
         Precondition: 
         Postcondition: 
      */
      void insertAt(int location, int insertItem);
  
      /* insertEnd Function ...
         Precondition: 
         Postcondition: 
      */
      void insertEnd(int insertItem);

      /* replaceAt Function ...
         Precondition: 
         Postcondition: 
      */
      void replaceAt(int location, int repItem);
  
      /* seqSearh Function ...
         Precondition: 
         Postcondition: 
      */
      int seqSearch(int searchItem) const;

}; 

#endif
