#include "omp.h"
#include "stdio.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>
#define NUM_THREADS 4
static long num_steps = 1000000;

void calcularPi(int ID, double* sum, int h){    
}
__global__ void
calcularPi(float *pi, int numElements, int operaciones)
{
	for(int j = 0; j < operaciones; j = j+4){
		int i = ((blockDim.x * blockIdx.x + threadIdx.x)*operaciones) + j;
	    pi+= 1.0/i;
        i +=2;
        pi -= 1.0/i;	
	}
    
}
void main()
{
  // declarar  la cantidad de hilos segun la gpu
  cudaError_t err = cudaSuccess;
	int dev = 0;
	cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	int threadsPerBlock = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	threadsPerBlock = threadsPerBlock*2;
  int blocksPerGrid =   deviceProp.multiProcessorCount;
  int numIt = 4000000000;
  int hilosTotales = blocksPerGrid*threadsPerBlock;
	int operacionPorHilo = numIt > hilosTotales ? (( numIt / hilosTotales )) + 1 ) : 1;
  float *pi;
  *pi = 0;
}