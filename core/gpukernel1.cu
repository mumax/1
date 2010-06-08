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
		if (repetition[i]<=1 && zero_pad[i]!=0){
			fprintf(stderr, "repetition[%d]= %d, while no periodicity is considered!", i, repetition[i]);
			fprintf(stderr, "repetition[%d], should be 1", i);
			return;
		}
	}
	
	int *Nkernel =  (int*)calloc(3, sizeof(int));
  Nkernel[X] = (1 + zero_pad[X]) * N0; 
  Nkernel[Y] = (1 + zero_pad[Y]) * N1; 
  Nkernel[Z] = (1 + zero_pad[Z]) * N2;
  int NkernelN = Nkernel[X] * Nkernel[Y] * Nkernel[Z];
  
	int NkernelStorageN = Nkernel[X] * Nkernel[Y] * (Nkernel[Z] + 2); 									///@todo probably best to define GPU_STRIDE_FLOAT
//	int NkernelStorageN = Nkernel[X] * Nkernel[Y] * (Nkernel[Z] + gpu_stride_float());  ///@todo probably best to define GPU_STRIDE_FLOAT
	
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


//	float cst = Ms/4/Pi/(float)NkernelN;         
	float cst = 1.0;												///@todo what is the normalized constant?
	gpu_init_and_FFT_Greens_kernel_elements(dev_kernel, Nkernel, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10, kernel_plan);
	
	return;
}


/// @todo argument defining which Greens function should be added
/// remark: number of FD cells in a dimension can not be odd if no zero padding!!
void gpu_init_and_FFT_Greens_kernel_elements(tensor *dev_kernel, int *Nkernel, float *FD_cell_size, float cst, int *repetition, float *dev_qd_P_10, float *dev_qd_W_10, gpu_plan3d_real_input* kernel_plan){

	int NkernelStorageN = 2*dev_kernel->size[1];

	float *host_temp = (float *)calloc(NkernelStorageN, sizeof(float));		// temp tensor on host for storage of each component in real + i*complex format
	float *dev_temp = new_gpu_array(NkernelStorageN);		                  // temp tensor on device for storage of each component in real + i*complex format
	
	dim3 gridsize(Nkernel[X]/2, Nkernel[Y]/2, 1);	///@todo generalize!
  dim3 blocksize(Nkernel[Z]/2, 1, 1);
  gpu_checkconf(gridsize, blocksize);
	
  int threadsPerBlock = 512;
  int blocks = (NkernelStorageN/2) / threadsPerBlock;
  gpu_checkconf_int(blocks, threadsPerBlock);

	gpu_zero(dev_temp, NkernelStorageN);
	cudaThreadSynchronize();
	_gpu_init_Greens_kernel_elements<<<gridsize, blocksize>>>(dev_temp, Nkernel, co1, co2, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10);
	cudaThreadSynchronize();
	
  memcpy_from_gpu(dev_temp, host_temp, NkernelStorageN);
	for (int i=0; i<Nkernel[X]; i++)
		for (int j=0; j<Nkernel[Y]; j++){
			for (int k=0; k<Nkernel[Z]; k++){
				fprintf(stderr, "%e ", host_temp[i*Nkernel[Y]*Nkernel[Z] + j*Nkernel[Z] + k);
			}
			fprintf(stderr, "\n");
		}
// 	int rang1=0;
// 	for (int co1=0; co1<3; co1++){
// 		for (int co2=co1; co2<3; co2++){
// 			gpu_zero(dev_temp, NkernelStorageN);
// 			cudaThreadSynchronize();
// 		  _gpu_init_Greens_kernel_elements<<<gridsize, blocksize>>>(dev_temp, Nkernel, co1, co2, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10);
// 			cudaThreadSynchronize();
// 		  gpu_plan3d_real_input_forward(kernel_plan, dev_temp);
// 			cudaThreadSynchronize();
// 			_gpu_extract_real_parts<<<blocks, threadsPerBlock>>>(dev_kernel->list, dev_temp, rang1, NkernelStorageN/2);
// 			cudaThreadSynchronize();
// 			rang1++;
// 		}
// 	}



	free (host_temp);
	free (dev_temp);   ///> @todo check how to free a float array on device!
	
	return;
}



__global__ void _gpu_init_Greens_kernel_elements(float *dev_temp, int *Nkernel, int co1, int co2, float *FD_cell_size, float cst, int *repetition, float *dev_qd_P_10, float *dev_qd_W_10){
   
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;

		int N12 = Nkernel[Y] * (Nkernel[Z] + 2);			///@todo aanpassen!!
		int N2 = Nkernel[Z] + 2;											///@todo aanpassen!!
//		int N12 = Nkernel[Y] * (Nkernel[Z] + gpu_stride_float());			///@todo aanpassen!!
//		int N2 = Nkernel[Z] + gpu_stride_float();											///@todo aanpassen!!

		
			dev_temp[             i*N12 +              j*N2 +            k] = _gpu_get_Greens_element(Nkernel, co1, co2,  i,  j,  k, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10);
		if (i>0)
			dev_temp[(Nkernel[X]-i)*N12 +              j*N2 +            k] = _gpu_get_Greens_element(Nkernel, co1, co2, -i,  j,  k, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10);
		if (j>0)
			dev_temp[             i*N12 + (Nkernel[Y]-j)*N2 +            k] = _gpu_get_Greens_element(Nkernel, co1, co2,  i, -j,  k, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10);
		if (k>0) 
			dev_temp[             i*N12 +              j*N2 + Nkernel[Z]-k] = _gpu_get_Greens_element(Nkernel, co1, co2,  i,  j, -k, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10);
		if (i>0 && j>0) 
			dev_temp[(Nkernel[X]-i)*N12 + (Nkernel[Y]-j)*N2 +            k] = _gpu_get_Greens_element(Nkernel, co1, co2, -i, -j,  k, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10);
		if (i>0 && k>0) 
			dev_temp[(Nkernel[X]-i)*N12 +              j*N2 + Nkernel[Z]-k] = _gpu_get_Greens_element(Nkernel, co1, co2, -i,  j, -k, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10);
		if (j>0 && k>0) 
			dev_temp[             i*N12 + (Nkernel[Y]-j)*N2 + Nkernel[Z]-k] = _gpu_get_Greens_element(Nkernel, co1, co2,  i, -j, -k, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10);
		if (i>0 && j>0 && k>0) 
			dev_temp[(Nkernel[X]-i)*N12 + (Nkernel[Y]-j)*N2 + Nkernel[Z]-k] = _gpu_get_Greens_element(Nkernel, co1, co2, -i, -j, -k, FD_cell_size, cst, repetition, dev_qd_P_10, dev_qd_W_10);
}

__device__ float _gpu_get_Greens_element(int *Nkernel, int co1, int co2, int a, int b, int c, float *FD_cell_size, float cst, int *repetition, float *dev_qd_P_10, float *dev_qd_W_10){

	float result = 0.0;
	float *dev_qd_P_10_X = &dev_qd_P_10[X];
	float *dev_qd_P_10_Y = &dev_qd_P_10[Y];
	float *dev_qd_P_10_Z = &dev_qd_P_10[Z];

	if (co1==0 && co2==0){

		for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
		for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
		for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

			int i = a + cnta*Nkernel[X]/2;
			int j = b + cntb*Nkernel[Y]/2;
			int k = c + cntc*Nkernel[Z]/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float x1 = (i + 0.5f) * FD_cell_size[X];
				float x2 = (i - 0.5f) * FD_cell_size[X];
				for (int cnt2=0; cnt2<10; cnt2++){
					float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2];
					for (int cnt3=0; cnt3<10; cnt3++){
						float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3];
						result += FD_cell_size[Y] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
							( x1*__powf(x1*x1+y*y+z*z, -1.5f) - x2*__powf(x2*x2+y*y+z*z, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
				result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
									(1.0f/ __powf(r2,1.5f) - 3.0f* (i*FD_cell_size[X]) * (i*FD_cell_size[X]) * __powf(r2,-2.5f));
			}
		}
		
	}
	
	if (co1==0 && co2==1){

		for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
		for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
		for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

			int i = a + cnta*Nkernel[X]/2;
			int j = b + cntb*Nkernel[Y]/2;
			int k = c + cntc*Nkernel[Z]/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float x1 = (i + 0.5f) * FD_cell_size[X];
				float x2 = (i - 0.5f) * FD_cell_size[X];
				for (int cnt2=0; cnt2<10; cnt2++){
					float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2];
					for (int cnt3=0; cnt3<10; cnt3++){
						float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3];
						result += FD_cell_size[Y] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
							( y*__powf(x1*x1+y*y+z*z, -1.5f) - y*__powf(x2*x2+y*y+z*z, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
				result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
									(- 3.0f* (i*FD_cell_size[X]) * (j*FD_cell_size[Y]) * __powf(r2,-2.5f));
			}
		}

	}

	if (co1==0 && co2==2){

		for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
		for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
		for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

			int i = a + cnta*Nkernel[X]/2;
			int j = b + cntb*Nkernel[Y]/2;
			int k = c + cntc*Nkernel[Z]/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float x1 = (i + 0.5f) * FD_cell_size[X];
				float x2 = (i - 0.5f) * FD_cell_size[X];
				for (int cnt2=0; cnt2<10; cnt2++){
					float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2];
					for (int cnt3=0; cnt3<10; cnt3++){
						float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3];
						result += FD_cell_size[Y] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt2] * dev_qd_W_10[cnt3] *
							( z*__powf(x1*x1+y*y+z*z, -1.5f) - z*__powf(x2*x2+y*y+z*z, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
				result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
									(- 3.0f* (i*FD_cell_size[X]) * (k*FD_cell_size[Y]) * __powf(r2,-2.5f));
			}
		}
	
	}
	
	if (co1==1 && co2==1){
	
		for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
		for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
		for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

			int i = a + cnta*Nkernel[X]/2;
			int j = b + cntb*Nkernel[Y]/2;
			int k = c + cntc*Nkernel[Z]/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float y1 = (j + 0.5f) * FD_cell_size[Y];
				float y2 = (j - 0.5f) * FD_cell_size[Y];
				for (int cnt1=0; cnt1<10; cnt1++){
					float x = i * FD_cell_size[X] + dev_qd_P_10_X[cnt1];
					for (int cnt3=0; cnt3<10; cnt3++){
						float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3];
						result += FD_cell_size[X] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
							( y1*__powf(x*x+y1*y1+z*z, -1.5f) - y2*__powf(x*x+y2*y2+z*z, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
				result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
									(1.0f/ __powf(r2,1.5f) - 3.0f* (j*FD_cell_size[Y]) * (j*FD_cell_size[Y]) * __powf(r2,-2.5f));
			}
		}

	}

	if (co1==1 && co2==2){
	
		for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
		for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
		for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

			int i = a + cnta*Nkernel[X]/2;
			int j = b + cntb*Nkernel[Y]/2;
			int k = c + cntc*Nkernel[Z]/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float y1 = (j + 0.5f) * FD_cell_size[Y];
				float y2 = (j - 0.5f) * FD_cell_size[Y];
				for (int cnt1=0; cnt1<10; cnt1++){
					float x = i * FD_cell_size[X] + dev_qd_P_10_X[cnt1];
					for (int cnt3=0; cnt3<10; cnt3++){
						float z = k * FD_cell_size[Z] + dev_qd_P_10_Z[cnt3];
						result += FD_cell_size[X] * FD_cell_size[Z] / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt3] *
							( z*__powf(x*x+y1*y1+z*z, -1.5f) - z*__powf(x*x+y2*y2+z*z, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
				result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
									( - 3.0f* (j*FD_cell_size[Y]) * (k*FD_cell_size[Z]) * __powf(r2,-2.5f));
			}
		}

	}

	if (co1==2 && co2==2){
	
		for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
		for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
		for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

			int i = a + cnta*Nkernel[X]/2;
			int j = b + cntb*Nkernel[Y]/2;
			int k = c + cntc*Nkernel[Z]/2;
			int r2_int = i*i+j*j+k*k;

			if (r2_int<400){
				float z1 = (k + 0.5f) * FD_cell_size[Z];
				float z2 = (k - 0.5f) * FD_cell_size[Z];
				for (int cnt1=0; cnt1<10; cnt1++){
					float x = i * FD_cell_size[X] + dev_qd_P_10_X[cnt1];
					for (int cnt2=0; cnt2<10; cnt2++){
						float y = j * FD_cell_size[Y] + dev_qd_P_10_Y[cnt2];
						result += FD_cell_size[X] * FD_cell_size[Y] / 4.0f * dev_qd_W_10[cnt1] * dev_qd_W_10[cnt2] *
							( z1*__powf(x*x+y*y+z1*z1, -1.5f) - z2*__powf(x*x+y*y+z2*z2, -1.5f));
					}
				}
			}
			else{
				float r2 = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
				result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
									(1.0f/ __powf(r2,1.5f) - 3.0f* (k*FD_cell_size[Z]) * (k*FD_cell_size[Z]) * __powf(r2,-2.5f));
			}
		}

	}
	
	return( cst*result );
	
}

__global__ void _gpu_extract_real_parts(float *dev_kernel_array, float *dev_temp, int rang1, int size1){

  int e = ((blockIdx.x * blockDim.x) + threadIdx.x);

	dev_kernel_array[rang1*size1 + e] = dev_temp[2*e];

	return;
}


// Gauss quadrature functions ___________e_____________________________________
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

