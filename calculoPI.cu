#include "omp.h"
#include "stdio.h"
#include <math.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>


__global__ void
calcularPi( float *sum, int operaciones)
{
  int i = ((blockDim.x * blockIdx.x + threadIdx.x));
  sum[i] = 0;
	for(int j = 0; j < operaciones; j++){
    float aumento = (float)(2/((4*(i + j) + 1)*(4*(i + j) + 3)));
    sum[i] += aumento ;
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
  int hilosTotales = blocksPerGrid*threadsPerBlock;
  int operacionPorHilo;
  size_t size_pi = sizeof(float) * hilosTotales;
  operacionPorHilo = (numIt > hilosTotales ) ? (int)(ceil(numIt/hilosTotales) ) : 1;
  float h_pi = 0.0;
  float *h_sum = (float*)malloc(size_pi);
  float *d_sum = NULL;
  err = cudaMalloc((void **)&d_sum, size_pi);
  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to allocate device d_sum (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }
  err = cudaMemcpy(d_sum, h_sum, size_pi, cudaMemcpyHostToDevice);

  

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector pi from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  printf("Operaciones por Hilo %d\n",operacionPorHilo);
  calcularPi<<<blocksPerGrid, threadsPerBlock>>>(d_sum, operacionPorHilo);
  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch calcularPi kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_sum, d_sum, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        
        fprintf(stderr, "Failed to copy h_pi from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    for(int i = 0 ; i < hilosTotales; i ++){
        h_pi += h_sum[i];
    }

    printf("valor de pi %f\n", (h_pi)*4);
    return 0;

}