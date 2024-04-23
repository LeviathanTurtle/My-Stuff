/*
Name: Justin Chen, William Wadsworth, Name
Class: csc-2710
Date: 3/4/2022
Program 2 creating a bunch of sorting algo.


Things to work on
1. get the counters for mergesort, heapsort, and quicksort   done?
2. get the times set for the sorts.                          done?
3. fix selection sort.
*/

#include <iostream>
#include <iomanip>
#include <sys/time.h>
using namespace std;

//global to show first 10 val in array
int p = 10;
//#define TEST 10



void swap(int *xp, int *yp) 
{ 
    int temp = *xp; 
    *xp = *yp; 
    *yp = temp; 
} 
  
void displayArray(int arr[], int size)//function for displaying the array
{
 int i;
 for(i = 0; i<size; i++)
   cout<<arr[i]<<" ";

 cout << endl;
}
//end if displaying function 

void selectionSort(int arr[], int size, int cnt) //not working right now
{
   int i, j, imin;
   for(i = 0; i<size-1; i++) 
   {
      imin = i;   //get index of minimum data
      for(j = i+1; j<size; j++)
         if(arr[j] < arr[imin])
         {
            cnt++;
            imin = j;
         }//placing in correct position
         swap(arr[i], arr[imin]);
   }
    cout << "the array is now sorted" << endl;
    cout << "this is the amount of swaps used: " << cnt << endl;
    displayArray(arr, p);
}

void exchangeSort(int arr[], int n, int cnt)
{
   int temp;
   for(int i=0; i < n-1; i++)
   {
      for(int j=i+1; j < n; j++)
      { 
        if(arr[j] < arr[i])
        {
            cnt++;
            temp = arr[i];
            arr[i] = arr[j];
            arr[j] = temp;
        }
      }
   }
    cout << "the array is now sorted" << endl;
    cout << "this is the amount of swaps used: " << cnt << endl;
    displayArray(arr, p);
}

void bubbleSort(int arr[], int n, int cnt) 
{ 
    int i, j; 
    for (i = 0; i < n-1; i++)     
      
    // Last i elements are already in place 
    for (j = 0; j < n-i-1; j++)
    {    
        cnt++; 
        if (arr[j] > arr[j+1]) 
            swap(&arr[j], &arr[j+1]);
    }         
    cout << "the array is now sorted" << endl;
    cout << "this is the amount of compares used: " << cnt << endl;
    displayArray(arr, p);
} 

void insertionSort(int arr[], int n, int cnt)
{
    int i, key, j;
    for (i = 1; i < n; i++)
    {
        key = arr[i];
        j = i - 1;
        cnt++;
        while (j >= 0 && arr[j] > key)
        {
            arr[j + 1] = arr[j];
            j = j - 1;
            cnt++;
        }
        arr[j + 1] = key;
    }
    
    cout << "the array is now sorted" << endl;
    cout << "this is the amount of compares used: " << cnt << endl;
    displayArray(arr, p);
}

void merge(int *arr, int low, int high, int mid, int& comp) //function for sorting each part of the array
{
    int i, j, k , c[100];
    i = low;
    k = low;
    j = mid + 1;

    while(i <= mid && j <= high)
    {
        comp += 2;
        if(arr[i] < arr[j])
        {
            c[k] = arr[i];
            k++;
            i++;
        }
        else
        {
            c[k] = arr[j];
            k++;
            j++;
        }
        comp++;
    }
    comp += 2;

    while(i <= mid)
    {
        comp++;
        c[k] = arr[i];
        k++;
        i++;
    }
    comp++;

    while(j <= high)
    {
        comp++;
        c[k] = arr[j]; 
        k++;
        j++;
    }
    comp++;
    
    for (i = low; i < k; i++, comp++)
    {
        //comp++;
        arr[i] = c[i];
    }
    comp++;
}
//end of merge sort function

void mergeSort(int *arr, int low, int high, int& comp)//fuction for dividing the array into parts
//this function calls in the sort function after it has been divided 
{
    int mid;
    if(low < high)
    {
        comp++;
        //divides the array at mid and sort independently using merge sort
        mid = (low+high)/2;
        mergeSort(arr, low, mid, comp);
        mergeSort(arr, mid+1, high, comp);
        //merge or conquer sorted arrays
        merge(arr, low, high, mid, comp);
    }
    comp++;
}
//end of dividing function

int partition(int arr[], int low, int high, int& comp) //partition function
{
    int pivot = arr[high];
    int i = (low - 1);
 
    for(int j = low; j <= high-1; j++, comp++)
    {
        //comp++;
        if(arr[j] <= pivot) //if current element is smaller than pivot, increment low element
        //swap elements at i and j
        {
            i++; //increments index of smaller element
            swap(&arr[i], &arr[j]);
        }
        comp++;
    }
    comp++;
    swap(&arr[i + 1], &arr[high]);
    return(i + 1);
}
//end of partition function

void quickSort(int arr[], int low, int high, int& comp) //quick sort function
{
    if(low < high)
    {
        //partition the array
        int pivot = partition(arr, low, high, comp);
  
        //sort the sub arrays independently
        quickSort(arr, low, pivot - 1, comp);
        quickSort(arr, pivot + 1, high, comp);
    }
    comp++;
}
//end of quick sort function

void heapify(int arr[], int n, int root, int& comp) //function to heapify the tree
{
    int largest = root; //root is the largest element
    int l = 2*root + 1; //left = 2*root + 1
    int r = 2*root + 2; //right = 2*root + 2

    if (l < n && arr[l] > arr[largest]) //if left child is larger than root
        largest = l;
    comp += 2;
 
    if (r < n && arr[r] > arr[largest]) //If right child is larger than largest
        largest = r;
    comp += 2;

    if(largest != root) //if largest is not root
    {
        swap(arr[root], arr[largest]);//swap root and largest
        heapify(arr, n, largest, comp);//Heapify sub-tree using recursion
    }
    comp++;
}
//End of heapify function

void heapSort(int arr[], int n, int& comp) //function implimenting Heap sort
{
    for (int i = n/2 - 1; i >= 0; i--, comp++) //build heap
    {
        //comp++;
        heapify(arr, n, i, comp);
    }
    comp++;
 
    for(int i = n-1; i >= 0; i--, comp++) //extracting elements from heap one by one
    {
        //comp++;
        swap(arr[0], arr[i]);
        heapify(arr, i, 0, comp);
    }
    comp++;
}
//End of Heap sort function

int main()
{
    int n = 100;
    int array1[100]; // almost sorted
    int array2[100]; // randomly sorted
    int array3[100]; // revere sorted
    int array4[100]; // duplications

    for(int i=0; i< n; i++)//puting values in array1
    {
        array1[i] = i+1;
    }

    for(int i=0; i<n; i++) //putting values in array2
    {
        array2[i] = rand()%1000;
    }

    int z= 0;
    for(int i=100; i>0; i--) //putting values in array3
    {
        array3[z] = i;
        z++;
    }
    
    for(int i=0; i<n; i++) //putting values in array4
    {
        array4[i] = rand()%15;
    }
    
    int x;
    cout << "the sorting algorthims are:" << endl;
    cout << "1 for selection sort" << endl;
    cout << "2 for exchange sort" << endl;
    cout << "3 for bubble sort" << endl;
    cout << "4 for insertion sort" << endl;
    cout << "5 for merge sort" << endl;
    cout << "6 for quicksort" << endl;
    cout << "7 for heapsort" << endl;
    cout << "which sort do you want: ";
    cin >> x;

    int cnt = 0; //counter for array
    
    // TIME VARIABLES
    struct timeval startTime, stopTime;
    double diff;

    // COMPARISON COUNTERS
    int mComp = 0, hComp = 0, qComp = 0;

    if(x==1)// selection sort
    {
        gettimeofday(&startTime, NULL);
        cout << "Almost sorted array:" << endl;
        selectionSort(array1, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;



        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Randomly sorted array:" << endl;
        selectionSort(array2, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;


        
        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Reverse sorted array:" << endl;
        selectionSort(array3, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;



        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Duplicate sorted array:" << endl;
        selectionSort(array4, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;
    }
    else if(x == 2)// exchange sort
    {
        gettimeofday(&startTime, NULL);
        cout << "Almost sorted array:" << endl;
        exchangeSort(array1, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;



        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Randomly sorted array:" << endl;
        exchangeSort(array2, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;



        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Reverse sorted array:" << endl;
        exchangeSort(array3, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;



        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Duplicate sorted array:" << endl;
        exchangeSort(array4, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;
    }
    else if(x == 3) // bubble sort
    {
        gettimeofday(&startTime, NULL);
        cout << "Almost sorted array:" << endl;
        bubbleSort(array1, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;



        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Randomly sorted array:" << endl;
        bubbleSort(array2, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;



        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Reverse sorted array:" << endl;
        bubbleSort(array3, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;



        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Duplicate sorted array:" << endl;
        bubbleSort(array4, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;
    }
    else if(x == 4) // insertion sort
    {
        gettimeofday(&startTime, NULL);
        cout << "Almost sorted array:" << endl;
        insertionSort(array1, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;



        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Randomly sorted array:" << endl;
        insertionSort(array2, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;



        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Reverse sorted array:" << endl;
        insertionSort(array3, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;



        diff = 0;
        gettimeofday(&startTime, NULL);
        cout << "Duplicate sorted array:" << endl;
        insertionSort(array4, n, cnt);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        //diff += (stopTime.tv_usec - starTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << diff << "s" << endl;
    }
    else if(x == 5) // mergesort
    {
        gettimeofday(&startTime, NULL);
        cout << "Almost sorted array:" << endl;
        mergeSort(array1, 0, n-1, mComp);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << mComp << endl;



        diff = 0, mComp = 0;
        gettimeofday(&startTime, NULL);
        cout << "Randomly sorted array:" << endl;
        mergeSort(array2, 0, n-1, mComp);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << mComp << endl;



        diff = 0, mComp = 0;
        gettimeofday(&startTime, NULL);
        cout << "Reverse sorted array:" << endl;
        mergeSort(array3, 0, n-1, mComp);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << mComp << endl;



        diff = 0, mComp = 0;
        gettimeofday(&startTime, NULL);
        cout << "Duplicate sorted array:" << endl;
        mergeSort(array4, 0, n-1, mComp);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << mComp << endl;
    }
    else if(x == 6) //quicksort
    {
        gettimeofday(&startTime, NULL);
        cout << "Almost sorted array:" << endl;
        quickSort(array1, 0, n-1, qComp);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << qComp << endl;



        diff = 0, qComp = 0;
        gettimeofday(&startTime, NULL);
        cout << "Randomly sorted array:" << endl;
        quickSort(array2, 0, n-1, qComp);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << qComp << endl;



        diff = 0, qComp = 0;
        gettimeofday(&startTime, NULL);
        cout << "Reverse sorted array:" << endl;
        quickSort(array3, 0, n-1, qComp);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << qComp << endl;



        diff = 0, qComp = 0;
        gettimeofday(&startTime, NULL);
        cout << "Duplicate sorted array:" << endl;
        quickSort(array4, 0, n-1, qComp);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << qComp << endl;
    }
    else if(x == 7) //heapsort
    {
        gettimeofday(&startTime, NULL);
        cout << "Almost sorted array:" << endl;
        heapSort(array1, n, hComp);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << hComp << endl;



        diff = 0, hComp = 0;
        gettimeofday(&startTime, NULL);
        cout << "Randomly sorted array:" << endl;
        heapSort(array2, n, hComp);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << hComp << endl;



        diff = 0, hComp = 0;
        gettimeofday(&startTime, NULL);
        cout << "Reverse sorted array:" << endl;
        heapSort(array3, n, hComp);
        gettimeofday(&stopTime, NULL);

        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << hComp << endl;



        diff = 0, hComp = 0;
        gettimeofday(&startTime, NULL);
        cout << "Duplicate sorted array:" << endl;
        heapSort(array4, n, hComp);
        gettimeofday(&stopTime, NULL);
        
        diff = stopTime.tv_sec - startTime.tv_sec;
        diff += (stopTime.tv_usec - startTime.tv_usec) / 1000;
        cout << endl << endl << "Time: " << fixed << setprecision(10) << diff << "s" << endl;
        cout << "Number of compares: " << hComp << endl;
    }

    return 0;
}

