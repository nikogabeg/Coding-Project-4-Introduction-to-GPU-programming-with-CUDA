// Niko Galedo
// CSC 656
// Vector Addition on GPU instead of CPU


#include <iostream>
#include <math.h>


// this code came from the CUDA C Programming Guide from the Picking up the Threads section

__global__
void add(int n, float *x, float *y)
{
 int index = threadIdx.x;
  int stride = blockDim.x;
  for (int i = index; i < n; i += stride)
      y[i] = x[i] + y[i];
}
 
int main(void)
{
 // step 3 says to change the size of the arrays to 512M elements
 int N = 1<<29; // 512M  elements
 float *x, *y;
 
// Allocate Unified Memory – accessible from CPU or GPU
 cudaMallocManaged(&x, N*sizeof(float));
 cudaMallocManaged(&y, N*sizeof(float));
 
 // initialize x and y arrays on the host
 for (int i = 0; i < N; i++) {
   x[i] = 1.0f;
   y[i] = 2.0f;
 }
 
 // Launch kernel with 256 threads
 add<<<1, 256>>>(N, x, y);

 // Wait for GPU to finish before accessing on host
 cudaDeviceSynchronize();

 // Check for errors (all values should be 3.0f)
 float maxError = 0.0f;
 for (int i = 0; i < N; i++)
   maxError = fmax(maxError, fabs(y[i]-3.0f));
 std::cout << "Max error: " << maxError << std::endl;
 
 // Free memory
 cudaFree(x);
 cudaFree(y);
 
 return 0;
}