1.  What is the average time for twenty runs of the serial version of the code?

    After 20 runs (using size 50), the average time was 3.264518e-7 seconds.


2.  What is the average time for twenty runs of the parallel version of the code?

    After 20 runs (using size 50 with 10 threads) the average time was 0.0180256 seconds.


3.  Calculate the speedup of the parallel version. Is the parallel code significantly faster?

    No, the serial verion is .0150487449096 seconds faster.


4.  The Methodology section above described how you decompose the summation routine to parallelize
    it. Obviously, OpenMP did all the work for you. How many elements of the array do you think
    OpenMP assigned to each processor? Hint: have your code print the number of threads in the
    computation (the function omp_get_thread_num() returns the number of threads).

    Knowing that OpenMP divides the work equally among each processor and the size of the array (50) was even, I would say it assigns 5 elements per processor.


5.  Can the program be written using (max:reduction)? If so, write another version.

    Yes.
