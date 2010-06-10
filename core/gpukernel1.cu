#include "tensor.h"
#include "gpufft.h"
#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>
#include "gpukernel1.h"

#ifdef __cplusplus
extern "C" {
#endif


void gpu_init_Greens_kernel1(tensor* dev_kernel, int N0, int N1, int N2, int *zero_pad, int *repetition, float *FD_cell_size){

	for (int i=0; i<3; i++){
		if (repetition[i]>1 && zero_pad[i]!=0){
			fprintf(stderr, "repetition[%d]= %d, while no periodicity is considered!\n", i, repetition[i]);
			fprintf(stderr, "repetition[%d], should be zero\n", i);
			return;
		}
	}
	
	int *Nkernel =  (int*)calloc(3, sizeof(int));
  Nkernel[X] = (1 + zero_pad[X]) * N0; 
  Nkernel[Y] = (1 + zero_pad[Y]) * N1; 
  Nkernel[Z] = (1 + zero_pad[Z]) * N2;
  
	int NkernelStorageN = Nkernel[X] * Nkernel[Y] * gpu_pad_to_stride(Nkernel[Z]+2);
	
	dev_kernel = as_tensor(new_gpu_array(6*NkernelStorageN/2), 2, 6, NkernelStorageN/2);  // only real parts!!

	// initialization Gauss quadrature points for integrations and copy to gpu ______________________
	float *dev_qd_W_10 = new_gpu_array(10);
	float *dev_qd_P_10 = new_gpu_array(3*10);
	initialize_Gauss_quadrature_on_gpu(dev_qd_W_10, dev_qd_P_10, FD_cell_size);
// ______________________________________________________________________________________________
	
	
// Plan initialization for FFTs Greens kernel elements __________________________________________		
	int* zero_pad_kernel = (int*)calloc(3, sizeof(int));
	zero_pad_kernel[X] = zero_pad_kernel[Y] = zero_pad_kernel[Z] = 0; 
	gpu_plan3d_real_input* kernel_plan = new_gpu_plan3d_real_input( (1+zero_pad[X])*N0, (1+zero_pad[Y])*N1, (1+zero_pad[Z])* N2, zero_pad_kernel);
// ______________________________________________________________________________________________


	float cst = -1.0f/4.0f/3.14159265f;
	gpu_init_and_FFT_Greens_kernel_elements(dev_kernel, Nkernel, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10, kernel_plan);
	
	return;
}

/// @todo argument defining which Greens function should be added
/// remark: number of FD cells in a dimension can not be odd if no zero padding!!
void gpu_init_and_FFT_Greens_kernel_elements(tensor *dev_kernel, int *Nkernel, float *FD_cell_size, float cst, int *repetition, float *dev_qd_P_10, float *dev_qd_W_10, gpu_plan3d_real_input* kernel_plan){

  
 	int NkernelStorageN = 2*dev_kernel->size[1];
	
	float *dev_temp = new_gpu_array(NkernelStorageN);		                  	// temp tensor on device for storage of each component in real + i*complex format
 	
	dim3 gridsize(Nkernel[X]/2, Nkernel[Y]/2, 1);	///@todo generalize!
  dim3 blocksize(Nkernel[Z]/2, 1, 1);				    ///@todo aan te passen!!  GPU_STRIDE_FLOAT
  gpu_checkconf(gridsize, blocksize);

	int gridsize2, blocksize2;
	make1dconf(NkernelStorageN/2, &gridsize2, &blocksize2);

	
	float *host_temp = (float *)calloc(NkernelStorageN, sizeof(float));			// temp array on host for storage of each component in real + i*complex format in serie (only for debugging purposes)
	float *host_temp2 = (float *)calloc(NkernelStorageN/2, sizeof(float));	// temp array on host for storage of only the real components

	int testco1 = 0;
	int testco2 = 0;
	int testrang = 0;
	for (int i=0; i<testco1; i++)
		for (int j=i; j<testco2; j++)
			testrang ++;
	fprintf(stderr, "test co: %d, %d, testrang: %d\n\n", testco1, testco2, testrang);

	gpu_zero(dev_temp, NkernelStorageN);
	cudaThreadSynchronize();
	_gpu_init_Greens_kernel_elements<<<gridsize, blocksize>>>(dev_temp, Nkernel[X], Nkernel[Y], Nkernel[Z], testco1, testco2, FD_cell_size[X], FD_cell_size[Y], FD_cell_size[Z], cst, repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
	cudaThreadSynchronize();

  memcpy_from_gpu(dev_temp, host_temp, NkernelStorageN);
	cudaThreadSynchronize();
	fprintf(stderr, "\nkernel elements (untransformed), co: %d, %d:\n", testco1, testco2);
	for (int i=0; i<Nkernel[X]; i++){
		for (int j=0; j<Nkernel[Y]; j++){
			for (int k=0; k<gpu_pad_to_stride(Nkernel[Z]+2); k++){
				fprintf(stderr, "%e ", host_temp[i*Nkernel[Y]*gpu_pad_to_stride(Nkernel[Z]+2) + j*gpu_pad_to_stride(Nkernel[Z]+2) + k]);
			}
			fprintf(stderr, "\n");
		}
		fprintf(stderr, "\n");
	}
	
	gpu_plan3d_real_input_forward(kernel_plan, dev_temp);
	cudaThreadSynchronize();
	
  memcpy_from_gpu(dev_temp, host_temp, NkernelStorageN);
	cudaThreadSynchronize();
	fprintf(stderr, "\nkernel elements (transformed), co: %d, %d:\n", testco1, testco2);
	for (int i=0; i<Nkernel[X]; i++){
		for (int j=0; j<Nkernel[Y]; j++){
			for (int k=0; k<gpu_pad_to_stride(Nkernel[Z]+2); k++){
				fprintf(stderr, "%e ", host_temp[i*Nkernel[Y]*gpu_pad_to_stride(Nkernel[Z]+2) + j*gpu_pad_to_stride(Nkernel[Z]+2) + k]);
			}
			fprintf(stderr, "\n");
		}
		fprintf(stderr, "\n");
	}

	_gpu_extract_real_parts<<<gridsize2, blocksize2>>>(&dev_kernel->list[testrang*NkernelStorageN/2], dev_temp, 0, NkernelStorageN/2);
	cudaThreadSynchronize();
	fprintf(stderr, "\nkernel elements (transformed, real parts), co: %d, %d:\n", testco1, testco2);
  memcpy_from_gpu(&dev_kernel->list[testrang*NkernelStorageN/2], host_temp2, NkernelStorageN/2);
	cudaThreadSynchronize();
		for (int i=0; i<Nkernel[X]; i++){
		for (int j=0; j<Nkernel[Y]; j++){
			for (int k=0; k<gpu_pad_to_stride(Nkernel[Z]+2)/2; k++){
				fprintf(stderr, "%e ", host_temp2[i*Nkernel[Y]*gpu_pad_to_stride(Nkernel[Z]+2)/2 + j*gpu_pad_to_stride(Nkernel[Z]+2)/2 + k]);
			}
			fprintf(stderr, "\n");
		}
		fprintf(stderr, "\n");
	}

	
	int rang1=0;
	for (int co1=0; co1<3; co1++){
		for (int co2=co1; co2<3; co2++){
			gpu_zero(dev_temp, NkernelStorageN);
			cudaThreadSynchronize();
		  _gpu_init_Greens_kernel_elements<<<gridsize, blocksize>>>(dev_temp, Nkernel[X], Nkernel[Y], Nkernel[Z], co1, co2, FD_cell_size[X], FD_cell_size[Y], FD_cell_size[Z], cst, repetition[X], repetition[Y], repetition[Z], dev_qd_P_10, dev_qd_W_10);
			cudaThreadSynchronize();
		  gpu_plan3d_real_input_forward(kernel_plan, dev_temp);
			cudaThreadSynchronize();
			_gpu_extract_real_parts<<<gridsize2, blocksize2>>>(&dev_kernel->list[rang1*NkernelStorageN/2], dev_temp, rang1, NkernelStorageN/2);
			cudaThreadSynchronize();
			rang1++;
		}
	}

	cudaFree (dev_temp);
	
	return;
}



__global__ void _gpu_init_Greens_kernel_elements(float *dev_temp, int Nkernel_X, int Nkernel_Y, int Nkernel_Z, int co1, int co2, float FD_cell_size_X, float FD_cell_size_Y, float FD_cell_size_Z, float cst, int repetition_X, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){
   
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;

//		int N2 = gpu_pad_to_stride(Nkernel_Z+2);
		int N2 = Nkernel_Z+2;     ///@todo: a gpu_pad_to_stride() function should also be executable on gpu
		int N12 = Nkernel_Y * N2;

			dev_temp[            i*N12 +             j*N2 +           k] = _gpu_get_Greens_element(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2,  i,  j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, cst, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
		if (i>0)
			dev_temp[(Nkernel_X-i)*N12 +             j*N2 +           k] = _gpu_get_Greens_element(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2, -i,  j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, cst, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
		if (j>0)
			dev_temp[            i*N12 + (Nkernel_Y-j)*N2 +           k] = _gpu_get_Greens_element(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2,  i, -j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, cst, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
		if (k>0) 
			dev_temp[            i*N12 +             j*N2 + Nkernel_Z-k] = _gpu_get_Greens_element(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2,  i,  j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, cst, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
		if (i>0 && j>0)
			dev_temp[(Nkernel_X-i)*N12 + (Nkernel_Y-j)*N2 +           k] = _gpu_get_Greens_element(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2, -i, -j,  k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, cst, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
		if (i>0 && k>0) 
			dev_temp[(Nkernel_X-i)*N12 +             j*N2 + Nkernel_Z-k] = _gpu_get_Greens_element(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2, -i,  j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, cst, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
		if (j>0 && k>0) 
			dev_temp[            i*N12 + (Nkernel_Y-j)*N2 + Nkernel_Z-k] = _gpu_get_Greens_element(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2,  i, -j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, cst, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);
		if (i>0 && j>0 && k>0) 
			dev_temp[(Nkernel_X-i)*N12 + (Nkernel_Y-j)*N2 + Nkernel_Z-k] = _gpu_get_Greens_element(Nkernel_X, Nkernel_Y, Nkernel_Z, co1, co2, -i, -j, -k, FD_cell_size_X, FD_cell_size_Y, FD_cell_size_Z, cst, repetition_X, repetition_Y, repetition_Z, dev_qd_P_10, dev_qd_W_10);

	return;
}

__device__ float _gpu_get_Greens_element(int Nkernel_X, int Nkernel_Y, int Nkernel_Z, int co1, int co2, int a, int b, int c, float FD_cell_size_X, float FD_cell_size_Y, float FD_cell_size_Z, float cst, int repetition_X, int repetition_Y, int repetition_Z, float *dev_qd_P_10, float *dev_qd_W_10){

	float result = 0.0f;
	float *dev_qd_P_10_X = &dev_qd_P_10[X];
	float *dev_qd_P_10_Y = &dev_qd_P_10[Y];
	float *dev_qd_P_10_Z = &dev_qd_P_10[Z];
	
	if (co1==0 && co2==0){
		for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
		for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
		for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

			int i = a + cnta*Nkernel_X/2;
			int j = b + cntb*Nkernel_Y/2;
			int k = c + cntc*Nkernel_Z/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float x1 = (i + 0.5f) * FD_cell_size_X;
				float x2 = (i - 0.5f) * FD_cell_size_X;
				for (int cnt2=0; cnt2<10; cnt2++){
					float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
					for (int cnt3=0; cnt3<10; cnt3++){
						float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
						result += FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
							( x1*__powf(x1*x1+y*y+z*z, -1.5f) - x2*__powf(x2*x2+y*y+z*z, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
				result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z *
									(1.0f/ __powf(r2,1.5f) - 3.0f* (i*FD_cell_size_X) * (i*FD_cell_size_X) * __powf(r2,-2.5f));
			}
		}
		result *= cst;
		if (a== 0 && b== 0 && c== 0) 	result -= 2.0f/FD_cell_size_X/FD_cell_size_X + 2.0f/FD_cell_size_Y/FD_cell_size_Y + 2.0f/FD_cell_size_Z/FD_cell_size_Z;
		if (a== 1 && b== 0 && c== 0)	result += 2.0f/FD_cell_size_X/FD_cell_size_X;
		if (a==-1 && b== 1 && c== 0)	result += 2.0f/FD_cell_size_X/FD_cell_size_X;
		if (a== 0 && b== 0 && c== 0)	result += 2.0f/FD_cell_size_Y/FD_cell_size_Y;
		if (a== 0 && b==-1 && c== 0)	result += 2.0f/FD_cell_size_Y/FD_cell_size_Y;
		if (a== 0 && b== 0 && c== 1)	result += 2.0f/FD_cell_size_Z/FD_cell_size_Z;
		if (a== 0 && b== 0 && c==-1)	result += 2.0f/FD_cell_size_Z/FD_cell_size_Z;
	}
	
	if (co1==0 && co2==1){
		for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
		for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
		for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

			int i = a + cnta*Nkernel_X/2;
			int j = b + cntb*Nkernel_Y/2;
			int k = c + cntc*Nkernel_Z/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float x1 = (i + 0.5f) * FD_cell_size_X;
				float x2 = (i - 0.5f) * FD_cell_size_X;
				for (int cnt2=0; cnt2<10; cnt2++){
					float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
					for (int cnt3=0; cnt3<10; cnt3++){
						float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
						result += FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
							( y*__powf(x1*x1+y*y+z*z, -1.5f) - y*__powf(x2*x2+y*y+z*z, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
				result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
									(- 3.0f* (i*FD_cell_size_X) * (j*FD_cell_size_Y) * __powf(r2,-2.5f));
			}
		}
		result *= cst;
	}

	if (co1==0 && co2==2){
		for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
		for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
		for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

			int i = a + cnta*Nkernel_X/2;
			int j = b + cntb*Nkernel_Y/2;
			int k = c + cntc*Nkernel_Z/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float x1 = (i + 0.5f) * FD_cell_size_X;
				float x2 = (i - 0.5f) * FD_cell_size_X;
				for (int cnt2=0; cnt2<10; cnt2++){
					float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
					for (int cnt3=0; cnt3<10; cnt3++){
						float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
						result += FD_cell_size_Y * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
							( z*__powf(x1*x1+y*y+z*z, -1.5f) - z*__powf(x2*x2+y*y+z*z, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
				result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
									(- 3.0f* (i*FD_cell_size_X) * (k*FD_cell_size_Y) * __powf(r2,-2.5f));
			}
		}
		result *= cst;
	}
	
	if (co1==1 && co2==1){
		for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
		for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
		for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

			int i = a + cnta*Nkernel_X/2;
			int j = b + cntb*Nkernel_Y/2;
			int k = c + cntc*Nkernel_Z/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float y1 = (j + 0.5f) * FD_cell_size_Y;
				float y2 = (j - 0.5f) * FD_cell_size_Y;
				for (int cnt1=0; cnt1<10; cnt1++){
					float x = i * FD_cell_size_X + dev_qd_P_10_X[cnt1];
					for (int cnt3=0; cnt3<10; cnt3++){
						float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
						result += FD_cell_size_X * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
							( y1*__powf(x*x+y1*y1+z*z, -1.5f) - y2*__powf(x*x+y2*y2+z*z, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
				result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
									(1.0f/ __powf(r2,1.5f) - 3.0f* (j*FD_cell_size_Y) * (j*FD_cell_size_Y) * __powf(r2,-2.5f));
			}
		}
		result *= cst;
		if (a== 0 && b== 0 && c== 0) 	result -= 2.0f/FD_cell_size_X/FD_cell_size_X + 2.0f/FD_cell_size_Y/FD_cell_size_Y + 2.0f/FD_cell_size_Z/FD_cell_size_Z;
		if (a== 1 && b== 0 && c== 0)	result += 2.0f/FD_cell_size_X/FD_cell_size_X;
		if (a==-1 && b== 1 && c== 0)	result += 2.0f/FD_cell_size_X/FD_cell_size_X;
		if (a== 0 && b== 0 && c== 0)	result += 2.0f/FD_cell_size_Y/FD_cell_size_Y;
		if (a== 0 && b==-1 && c== 0)	result += 2.0f/FD_cell_size_Y/FD_cell_size_Y;
		if (a== 0 && b== 0 && c== 1)	result += 2.0f/FD_cell_size_Z/FD_cell_size_Z;
		if (a== 0 && b== 0 && c==-1)	result += 2.0f/FD_cell_size_Z/FD_cell_size_Z;
	}

	if (co1==1 && co2==2){
		for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
		for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
		for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

			int i = a + cnta*Nkernel_X/2;
			int j = b + cntb*Nkernel_Y/2;
			int k = c + cntc*Nkernel_Z/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float y1 = (j + 0.5f) * FD_cell_size_Y;
				float y2 = (j - 0.5f) * FD_cell_size_Y;
				for (int cnt1=0; cnt1<10; cnt1++){
					float x = i * FD_cell_size_X + dev_qd_P_10_X[cnt1];
					for (int cnt3=0; cnt3<10; cnt3++){
						float z = k * FD_cell_size_Z + dev_qd_P_10_Z[cnt3];
						result += FD_cell_size_X * FD_cell_size_Z / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
							( z*__powf(x*x+y1*y1+z*z, -1.5f) - z*__powf(x*x+y2*y2+z*z, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
				result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
									( - 3.0f* (j*FD_cell_size_Y) * (k*FD_cell_size_Z) * __powf(r2,-2.5f));
			}
		}
		result *= cst;
	}

	if (co1==2 && co2==2){
		for(int cnta=-repetition_X; cnta<=repetition_X; cnta++)
		for(int cntb=-repetition_Y; cntb<=repetition_Y; cntb++)
		for(int cntc=-repetition_Z; cntc<=repetition_Z; cntc++){

			int i = a + cnta*Nkernel_X/2;
			int j = b + cntb*Nkernel_Y/2;
			int k = c + cntc*Nkernel_Z/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float z1 = (k + 0.5f) * FD_cell_size_Z;
				float z2 = (k - 0.5f) * FD_cell_size_Z;
				for (int cnt1=0; cnt1<10; cnt1++){
					float x = i * FD_cell_size_X + dev_qd_P_10_X[cnt1];
					for (int cnt2=0; cnt2<10; cnt2++){
						float y = j * FD_cell_size_Y + dev_qd_P_10_Y[cnt2];
						result += FD_cell_size_X * FD_cell_size_Y / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] *
							( z1*__powf(x*x+y*y+z1*z1, -1.5f) - z2*__powf(x*x+y*y+z2*z2, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size_X)*(i*FD_cell_size_X) + (j*FD_cell_size_Y)*(j*FD_cell_size_Y) + (k*FD_cell_size_Z)*(k*FD_cell_size_Z);
				result += FD_cell_size_X * FD_cell_size_Y * FD_cell_size_Z * 
									(1.0f/ __powf(r2,1.5f) - 3.0f* (k*FD_cell_size_Z) * (k*FD_cell_size_Z) * __powf(r2,-2.5f));
			}
		}
		result *= cst;
		if (a== 0 && b== 0 && c== 0) 	result -= 2.0f/FD_cell_size_X/FD_cell_size_X + 2.0f/FD_cell_size_Y/FD_cell_size_Y + 2.0f/FD_cell_size_Z/FD_cell_size_Z;
		if (a== 1 && b== 0 && c== 0)	result += 2.0f/FD_cell_size_X/FD_cell_size_X;
		if (a==-1 && b== 1 && c== 0)	result += 2.0f/FD_cell_size_X/FD_cell_size_X;
		if (a== 0 && b== 0 && c== 0)	result += 2.0f/FD_cell_size_Y/FD_cell_size_Y;
		if (a== 0 && b==-1 && c== 0)	result += 2.0f/FD_cell_size_Y/FD_cell_size_Y;
		if (a== 0 && b== 0 && c== 1)	result += 2.0f/FD_cell_size_Z/FD_cell_size_Z;
		if (a== 0 && b== 0 && c==-1)	result += 2.0f/FD_cell_size_Z/FD_cell_size_Z;
	}
	
	return( result );
	
}

__global__ void _gpu_extract_real_parts(float *dev_kernel_array, float *dev_temp, int rang1, int size1){

  int e = ((blockIdx.x * blockDim.x) + threadIdx.x);

	dev_kernel_array[rang1*size1 + e] = dev_temp[2*e];

	return;
}


// Gauss quadrature functions ________________________________________________
void initialize_Gauss_quadrature_on_gpu(float *dev_qd_W_10, float *dev_qd_P_10, float *FD_cell_size){

	float *std_qd_P_10 = (float*) calloc(10, sizeof(float));
	std_qd_P_10[0] = -0.97390652851717197f;
	std_qd_P_10[1] = -0.86506336668898498f;
	std_qd_P_10[2] = -0.67940956829902399f;
	std_qd_P_10[3] = -0.43339539412924699f;
	std_qd_P_10[4] = -0.14887433898163099f;
	std_qd_P_10[5] = -std_qd_P_10[4];
	std_qd_P_10[6] = -std_qd_P_10[3];
	std_qd_P_10[7] = -std_qd_P_10[2];
	std_qd_P_10[8] = -std_qd_P_10[1];
	std_qd_P_10[9] = -std_qd_P_10[0];
	float *host_qd_W_10 = (float*)calloc(10, sizeof(float));
	host_qd_W_10[0] = host_qd_W_10[9] = 0.066671344308687999f;
	host_qd_W_10[1] = host_qd_W_10[8] = 0.149451349150581f;
	host_qd_W_10[2] = host_qd_W_10[7] = 0.21908636251598201f;
	host_qd_W_10[3] = host_qd_W_10[6] = 0.26926671930999602f;
	host_qd_W_10[4] = host_qd_W_10[5] = 0.29552422471475298f;

	float *host_qd_P_10 =  (float *) calloc (3*10, sizeof(float));
	get_Quad_Points(&host_qd_P_10[X*10], std_qd_P_10, 10, -0.5f*FD_cell_size[X], 0.5f*FD_cell_size[X]);
	get_Quad_Points(&host_qd_P_10[Y*10], std_qd_P_10, 10, -0.5f*FD_cell_size[Y], 0.5f*FD_cell_size[Y]);
	get_Quad_Points(&host_qd_P_10[Z*10], std_qd_P_10, 10, -0.5f*FD_cell_size[Z], 0.5f*FD_cell_size[Z]);

	memcpy_to_gpu (host_qd_W_10, dev_qd_W_10, 10);
	memcpy_to_gpu (host_qd_P_10, dev_qd_P_10, 3*10);

	free (std_qd_P_10);
	free (host_qd_P_10);
	free (host_qd_W_10);

	return;
}

void get_Quad_Points(float *gaussQP, float *stdGaussQP, int qOrder, double a, double b){
// get the quadrature points for integration between a and b
	int i;
	double A = (b-a)/2.0f; // coefficients for transformation x'= Ax+B
	double B = (a+b)/2.0f; // where x' is the new integration parameter

	gaussQP = (float *) calloc(qOrder, sizeof(float));

	for(i = 0; i < qOrder; i++)
		gaussQP[i] = A*stdGaussQP[i]+B;

	return;
}
// ___________________________________________________________________________


#ifdef __cplusplus
}
#endif



// remove the following if code is tested.

