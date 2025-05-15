// Niko Galedo
// CSC 656
// Vector Addition on CPU
// I followed the example from the CUDA C Programming Guide
// and modified it to run on the CPU with the addition of the
// timing code to measure the elapsed time for the vector addition.


#include <iostream>
#include <math.h>
#include <chrono>


// this code came from the CUDA C Programming Guide

// function to add the elements of two arrays
void add(int n, float *x, float *y)
{
 for (int i = 0; i < n; i++)
     y[i] = x[i] + y[i];
}
 
int main(void)
{
 // step 3 says to change the size of the arrays to 512M elements
 int N = 1<<29; // 512M  elements
 
 float *x = new float[N];
 float *y = new float[N];
 
 // initialize x and y arrays on the host
 for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }

 // added the timer from benchmark.cpp code from project 2
 // and the example from chrono timer cpp file 
 // which allowed me to implement the timer here:
  
 // Start timer
 auto start = std::chrono::high_resolution_clock::now();
 
 // Run kernel on 512M elements on the CPU
 add(N, x, y);

 // End timer
 auto end = std::chrono::high_resolution_clock::now();
 
 // Calculate elapsed time in milliseconds
 std::chrono::duration<double> elapsed = end - start;
 double elapsed_ms = elapsed.count() * 1000.0;

 // Print elapsed time
 std::cout << "Elapsed time for vector addition: " << elapsed_ms << " ms" << std::endl;

 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++)
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 std::cout << "Max error: " << maxError << std::endl;
 
 // Free memory
 delete [] x;
 delete [] y;
 
 return 0;
}