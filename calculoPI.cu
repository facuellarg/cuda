#include "omp.h"
#include "stdio.h"
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


__global__ void
calcularPi(float *pi, int operaciones)
{
  float i = ((blockDim.x * blockIdx.x + threadIdx.x)*operaciones);
  
	for(int j = 0; j < operaciones; j++){
    i = i + j;
    if(i < 10){
      printf("En %f valor %d\n",i,(2/((4*i + 1)*(4*i + 3))));
    }
    *pi = *pi + (2/((4*i + 1)*(4*i + 3)));
	}
    
}
int main(void)
{
  // declarar  la cantidad de hilos segun la gpu
  cudaError_t err = cudaSuccess;
  int dev = 0;
  size_t size = sizeof(float);
	cudaSetDevice(dev);
  cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	int threadsPerBlock = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
	threadsPerBlock = threadsPerBlock*2;
  int blocksPerGrid =   deviceProp.multiProcessorCount;
  float numIt = 4e9;
  printf("valor inicial%f\n", numIt);
  int hilosTotales = blocksPerGrid*threadsPerBlock;
  int operacionPorHilo;
  operacionPorHilo = (numIt > hilosTotales ) ? (int)(ceil(numIt/hilosTotales) ) : 1;
  float *h_pi = (float*)malloc(size);
  *h_pi = 0;
  float *d_pi = NULL;
  err = cudaMalloc((void **)&d_pi, size);
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device d_pi (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }


  err = cudaMemcpy(d_pi, h_pi, size, cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector pi from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  printf("Operaciones por Hilo %d\n",operacionPorHilo);
  calcularPi<<<blocksPerGrid, threadsPerBlock>>>(d_pi, operacionPorHilo);
  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch calcularPi kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_pi, d_pi, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        
        fprintf(stderr, "Failed to copy h_pi from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("valor de pi %f\n", (*h_pi)*4);
    return 0;

}