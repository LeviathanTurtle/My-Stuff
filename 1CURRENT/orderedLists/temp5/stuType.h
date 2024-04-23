/* William Wadsworth
 * CSC1720
 * stuType base class header and implementation
 */

#ifndef STU_TYPE
#define STU_TYPE

#include "orderedLinkedList.h"
using namespace std;

template <class Type>
class stuType: public orderedLinkedList<Type>
{
   protected:
      string first, last;
      double gpa;
      nodeType<Type> *next;
   public:
      stuType();
      // default constructor, default name is John Doe, gpa = 0

      double getGPA();
      // function to return the protected variable gpa

      double avgGPA();
      // function to go through the list and calculate the average gpa, and return it
};

template <class Type>
stuType<Type>::stuType()
{
    first = "John";
    last = "Doe";
    gpa = 0;
    next = nullptr;
}

template <class Type>
double stuType<Type>::getGPA()
{
    return gpa;
}

template <class Type>
double stuType<Type>::avgGPA()
{
    nodeType<Type> *ptr = this->first;
    double g;
    int count = 0;

    while (ptr != nullptr)
    {
        g += gpa;
        count++;
    }
    return gpa / count;
}

#endif