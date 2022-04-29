
#include <stdio.h>
#include <assert.h>
#include <stdlib.h>
#include <hip/hip_runtime.h> 
#include <time.h>
#include "CImg.h"
using namespace std;
/* Mandlebrot rendering function
   :inputs: width and height of domain, max_iterations
   :ouputs: 8biti unsigned character array containing mandlebrot image
*/
__global__ void render(unsigned char out[], const int width, const int height, const int max_iter) {


  int x_dim = blockIdx.x*blockDim.x + threadIdx.x;
  int y_dim = blockIdx.y*blockDim.y + threadIdx.y;
  // flatten the index.  
  int index = width*y_dim + x_dim;

  if(index >= width*height) return; 

  float x_origin = ((float) x_dim/width)*3.25 - 2; 
  float y_origin = ((float) y_dim/width)*2.5 - 1.25;

  float x = 0.0;
  float y = 0.0;

  int iteration = 0;
  //escape algorithm
  // Every thread will loop in this at most max_iter 
  while(x*x + y*y <= 4 && iteration < max_iter) {
    float xtemp = x*x - y*y + x_origin;
    y = 2*x*y + y_origin;
    x = xtemp;
    iteration++;
  }

  if(iteration == max_iter) {
    out[index] = 0;
  } else {
    out[index] = iteration;
  }
}

void mandelbrot(const int width, const int height, const int max_iter)
{
  
  size_t buffer_size = sizeof(char) * width * height;

  unsigned char *image; 
  hipMalloc(&image, buffer_size);

  unsigned char *host_image; 
  host_image = new unsigned char[width*height]; 

  dim3 block_Dim(16, 16, 1); // 16*16 threads 
  dim3 grid_Dim(width / block_Dim.x, height / block_Dim.y, 1); //Rest of the image

  render<<< grid_Dim, block_Dim >>>(image, width, height, max_iter);

  hipMemcpy(host_image, image, buffer_size, hipMemcpyDeviceToHost);

  cimg_library::CImg<unsigned char> img2(host_image, width, height);
  img2.save("output.bmp");

  hipFree(image);
  delete host_image;
}


/*main function */ 
int main() 
{
  clock_t tStart = clock();
  mandelbrot(1024, 1024, 256);
  printf("Time taken by 1024: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
  clock_t tStart = clock();
  mandelbrot(2048, 2048, 256);
  printf("Time taken by 2048: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
  clock_t tStart = clock();
  mandelbrot(4096, 4096, 256);
  printf("Time taken by 4096: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
  clock_t tStart = clock();
  mandelbrot(8192 8192, 256);
  printf("Time taken by 8192: %.2fs\n", (double)(clock() - tStart)/CLOCKS_PER_SEC);
  return 0;
}