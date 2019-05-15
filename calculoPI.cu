#include "omp.h"
#include "stdio.h"
#include <cuda_runtime.h>
#include <helper_cuda.h>


__global__ void
calcularPi(float *pi, int numElements, int operaciones)
{
	for(int j = 0; j < operaciones; j = j+4){
		int i = ((blockDim.x * blockIdx.x + threadIdx.x)*operaciones) + j;
	      *pi+= 1.0/i;
        i +=2;
        *pi -= 1.0/i;	
	}
    
}
int main(void)
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
  long numIt = 4000000000;
  int hilosTotales = blocksPerGrid*threadsPerBlock;
	int operacionPorHilo = numIt > hilosTotales ? (( numIt / hilosTotales )) + 1 ) : 1;
  float *h_pi, *d_pi;
  *h_pi = 0;
  *d_pi = 0;

  err = cudaMemcpy(d_pi, h_pi, sizeof(float), cudaMemcpyHostToDevice);

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to copy vector pi from host to device (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
  printf("Operaciones por Hilo %d\n",operacionPorHilo);
  calcularPi<<<blocksPerGrid, threadsPerBlock>>>(d_pi, numIt,operacionPorHilo);
  err = cudaGetLastError();

  if (err != cudaSuccess)
  {
      fprintf(stderr, "Failed to launch calcularPi kernel (error code %s)!\n", cudaGetErrorString(err));
      exit(EXIT_FAILURE);
  }

  printf("Copy output data from the CUDA device to the host memory\n");
    err = cudaMemcpy(h_pi, d_pi, sizeof(float), cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        
        fprintf(stderr, "Failed to copy h_pi from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    printf("valor de pi %f\n", *h_pi);
    return 0;

}