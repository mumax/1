/*
 * Example of 3D cuda FFT. Copies some random data to the GPU where it is transformed back and forth, and copied back to the main RAM. The FFT error is checked.
 */

#include <cufft.h>
#include <stdio.h>
#include <assert.h>

double sqr(double x){
  return x*x;
}

int main(){
 
 int N0 = 128, N1 = 128, N2 = 8;	// size of the data to be transformed
 int N = 2 * N0 * N1 * N2;		// size of complex array
 
 float* data_host, *data_dev;		// the original data, on the host (PC) and a copy on the device (GPU)
 float* transf_dev; 			// the transformed data, on the device
 float* transf2_host, *transf2_dev; 	// transformed+backtransformed data, on the device and a copy on host
 
 data_host = (float*)malloc(N * sizeof(float));		
 cudaMalloc((void**)&data_dev, N * sizeof(float));	
 transf2_host = (float*)malloc((N) * sizeof(float));
 cudaMalloc((void**)&transf_dev, (N) * sizeof(float));
 cudaMalloc((void**)&transf2_dev, N * sizeof(float));
 
 int i;
 double rmserror = 0;
 
 cufftHandle plan;			
 cufftPlan3d(&plan, N0, N1, N2, CUFFT_C2C);
    
 for(i=0; i<N; i++){				// make some data on the host
  data_host[i] = (rand() % 10000) / 10000.0;
 }
 
  cudaMemcpy(data_dev, data_host, N * sizeof(float), cudaMemcpyHostToDevice);	// copy data to the device
  
  cufftExecC2C(plan, (cufftComplex*)data_dev, (cufftComplex*)transf_dev, CUFFT_FORWARD);
  cufftExecC2C(plan, (cufftComplex*)transf_dev, (cufftComplex*)transf2_dev, CUFFT_INVERSE);

  
  cudaMemcpy(transf2_host, transf2_dev, N * sizeof(float), cudaMemcpyDeviceToHost); // copy back to host
 
  // check RMS error of transform+backtransform
  for(i=0; i<N; i++){
    //printf("%f\n", transf2_host[i] / N);
    rmserror += sqr(data_host[i] - transf2_host[i] / (N0 * N1 * N2) );	
    //assert(transf2_host[i] != 0.0);
  }
  rmserror = sqrt(rmserror);
  printf("FFT error: %lf\n", rmserror);
  assert(rmserror < 1E-3);

  // clean up:
 cufftDestroy(plan);
 cudaFree(data_dev);
 cudaFree(transf2_dev);
 cudaFree(transf_dev);
 free(data_host);
 free(transf2_host);
 
  return 0;
}