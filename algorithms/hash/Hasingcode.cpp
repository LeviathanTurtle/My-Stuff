#include<iostream>
//#include<stdio.h>
using namespace std;

void findFrequency(int A[], int n)
{
    int freq[n];
  
    for(int i = 0; i < n; i++)
        freq[i] = 0;

    for(int i = 0; i < n; i++)
        freq[A[i]]++;
  
    for (int i = 0; i < n; i++)
        if (freq[i])
            printf("%d appears %d times\n", i, freq[i]);
}

int main()
{
    int A[] = { 2, 3, 3, 2, 1, 21};
    int n = sizeof(A) / sizeof(A[0]);
 
    findFrequency(A, n);
    
    return 0;
}
