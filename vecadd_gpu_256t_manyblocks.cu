// Niko Galedo
// CSC 656
// Vector Addition on GPU instead of CPU


#include <iostream>
#include <math.h>


// this code came from the CUDA C Programming Guide for Out of the Blocks section

__global__
void add(int n, float *x, float *y)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
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
 
 /*
 From Cude website:
 Together, the blocks of parallel threads make up what is known as the grid. 
 Since I have N elements to process, and 256 threads per block, 
 I just need to calculate the number of blocks to get at least N threads. 
 I simply divide N by the block size.
 */
 int blockSize = 256;
 int numBlocks = (N + blockSize - 1) / blockSize;

 //Print number of blocks as required
 std::cout << "Using " << numBlocks << " thread blocks" << std::endl;

 add<<<numBlocks, blockSize>>>(N, x, y);

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