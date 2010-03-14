/*
* Example of 1D cuda FFT. Copies some random data to the GPU where it is transformed back and forth, and copied back to the main RAM. The FFT error is checked.
*/

#include <cufft.h>
#include <stdio.h>
#include <assert.h>

double sqr(double x){
  return x*x;
}

int main(){
  
  int N = 1024;	// size of the data to be transformed
  int BATCH = 1; // number of data sets to be transformed in one batch
  
  float* data_host, *data_dev;		// the original data, on the host (PC) and a copy on the device (GPU)
  float* transf_dev; 			// the transformed data, on the device
  float* transf2_host, *transf2_dev; 	// transformed+backtransformed data, on the device and a copy on host
  
  data_host = (float*)malloc(N * sizeof(float));		// real
  cudaMalloc((void**)&data_dev, N * sizeof(float));	
  transf2_host = (float*)malloc((N+2) * sizeof(float));	// complex, transformed, so 2 floats larger 
  cudaMalloc((void**)&transf_dev, (N+2) * sizeof(float));
  cudaMalloc((void**)&transf2_dev, (N) * sizeof(float));
  
  int i;
  double rmserror = 0;
  
  cufftHandle fw_plan, bw_plan;			
  cufftPlan1d(&fw_plan, N, CUFFT_R2C, BATCH);	// initialize the forward and backward plans
  cufftPlan1d(&bw_plan, N, CUFFT_C2R, BATCH);
  
  for(i=0; i<N; i++){				// make some data on the host
    data_host[i] = (rand() % 10000) / 10000.0;
  }
  
  cudaMemcpy(data_dev, data_host, N * sizeof(float), cudaMemcpyHostToDevice);	// copy data to the device
  
  cufftExecR2C(fw_plan, (cufftReal*)data_dev, (cufftComplex*)transf_dev);	// transform on device
  cufftExecC2R(bw_plan, (cufftComplex*)transf_dev, (cufftReal*)transf2_dev); 	// backtransform on device
  
  cudaMemcpy(transf2_host, transf2_dev, N * sizeof(float), cudaMemcpyDeviceToHost); // copy back to host
  
  // check RMS error of transform+backtransform
  for(i=0; i<N; i++){
    rmserror += sqr(data_host[i] - transf2_host[i] / N);	
    assert(transf2_host[i] != 0.0);
  }
  rmserror = sqrt(rmserror);
  printf("FFT error: %lf\n", rmserror);
  assert(rmserror < 1E-5);
  
  // clean up:
  cufftDestroy(fw_plan);
  cudaFree(data_dev);
  cudaFree(transf2_dev);
  cudaFree(transf_dev);
  free(data_host);
  free(transf2_host);
  
  return 0;
}