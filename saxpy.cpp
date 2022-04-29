#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <time.h>
#include <iostream>
#include <hip/hip_runtime.h>
using namespace std;
__global__ void saxpy(int n, float a, const float x[], float y[])
{
	int id = threadIdx.x + blockDim.x*blockIdx.x; /* Performing that for loop */ 
	
	if(id < n){
		 y[id] +=  a*x[id]; // y[id] = y[id] + a*x[id]; 
	} 
}

int main()
{   
    vecotr<int> Ns = {16, 128, 1024, 2048, 65536}；
    for (auto N : Ns){
        clock_t tStart = clock();
        //create pointers and device
        float *d_x, *d_y; 
        
        const float a = 2.0f;

        //allocate and initializing memory on host
        std::vector<float> x(N, 1.f);
        std::vector<float> y(N, 1.f);
        /*
            float *x, *y; 
            x = new float[N]; //C++
            (*float)Malloc(x, N*sizeof(float)); //C
        */
        //allocate our memory on GPU 
        hipMalloc(&d_x, N*sizeof(float));
        hipMalloc(&d_y, N*sizeof(float));
        
        //Memory Transfer! 
        hipMemcpy(d_x, x.data(), N*sizeof(float), hipMemcpyHostToDevice);
        hipMemcpy(d_y, y.data(), N*sizeof(float), hipMemcpyHostToDevice); 

        saxpy<<<1, 256>>>(N, a, d_x, d_y);

        hipMemcpy(y.data(), d_y, N*sizeof(float), hipMemcpyDeviceToHost);
        //std::cout<<"First Element of z = ax + y is " << y[0]<<std::endl; 
        hipFree(d_x);
        hipFree(d_y);
        //std::cout<<"Done!"<<std::endl;  
        printf("Time taken by %d: %.2fs\n", N, (double)(clock() - tStart)/CLOCKS_PER_SEC);
    }
    return 0;
}