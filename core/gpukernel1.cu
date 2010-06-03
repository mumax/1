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


void gpu_init_Greens_kernel1(tensor* kernel, int N0, int N1, int N2, int *zero_pad, int *repetition, float *FD_cell_size){

	for (int i=0; i<3; i++){
		if (repetition[i]=<1 && zero_pad[i]!=0){
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
  
	int *NkernelStorage =  (int*)calloc(3, sizeof(int));
  NkernelStorage[X] = Nkernel[X]; 
  NkernelStorage[Y] = Nkernel[Y]; 
//  NkernelStorage[Z] = Nkernel[Z] + gpu_stride_float();   ///@todo aanpassen!!;
  NkernelStorage[Z] = Nkernel[Z] + 2;
	int NkernelStorageN = NkernelStorage[X] * NkernelStorage[Y] * NkernelStorage[Z];

// temp tensor for storage of each component in real + i*complex format _________________________	
	tensor *temp = new_tensor(3, NkernelStorage[X], NkernelStorage[Y], NkernelStorage[Z]);
// ______________________________________________________________________________________________
	
// initialization Gauss quadrature points for integrations ______________________________________
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
	float *qd_W_10 = (float*)calloc(10, sizeof(float));
	qd_W_10[0] = qd_W_10[9] = 0.066671344308687999f;
	qd_W_10[1] = qd_W_10[8] = 0.149451349150581f;
	qd_W_10[2] = qd_W_10[7] = 0.21908636251598201f;
	qd_W_10[3] = qd_W_10[6] = 0.26926671930999602f;
	qd_W_10[4] = qd_W_10[5] = 0.29552422471475298f;
	
	float **qd_P_10 =  (float **) calloc (3, sizeof(float **));
	get_Quad_Points(qd_P_10[X], std_qd_P_10, 10, -0.5f*FD_cell_size[X], 0.5f*FD_cell_size[X]);
	get_Quad_Points(qd_P_10[Y], std_qd_P_10, 10, -0.5f*FD_cell_size[Y], 0.5f*FD_cell_size[Y]);
	get_Quad_Points(qd_P_10[Z], std_qd_P_10, 10, -0.5f*FD_cell_size[Z], 0.5f*FD_cell_size[Z]);
	free (std_qd_P_10);
// ______________________________________________________________________________________________
	
	
// Plan initialization for FFTs Greens kernel elements __________________________________________		
	int* zero_pad_kernel = (int*)calloc(3, sizeof(int));
	zero_pad_kernel[X] = zero_pad_kernel[Y] = zero_pad_kernel[Z] = 0; 
	gpu_plan3d_real_input* kernel_plan = new_gpu_plan3d_real_input( (1+zero_pad[X])*N0, (1+zero_pad[Y])*N1, (1+zero_pad[Z])* N2, zero_pad_kernel);
// ______________________________________________________________________________________________


//	float cst = Ms/4/Pi/(float)NkernelN;         
	float cst = 1.0;												///@todo what is the normalized constant?
	gpu_init_and_FFT_Greens_kernel_elements(temp, Nkernel, FD_cell_size, cst, repetition, qd_P_10, qd_W_10);
	
	return();
}


/// @todo argument defining which Greens function should be added
/// remark: number of FD cells in a dimension can not be odd if no zero padding!!
void gpu_init_and_FFT_Greens_kernel_elements(tensor *temp, int *Nkernel, float *FD_cell_size, float cst, int *repetition, float **qd_P_10, float *qd_W_10){

	dim3 gridsize(Nkernel[X]/2, Nkernel[Y]/2, 1);	///@todo generalize!
  dim3 blocksize(Nkernel[Z]/2, 1, 1);
  gpu_checkconf(gridsize, blocksize);

	for (int i=0; i<3; i++){
		for (int j=i; j<3; j++){
			// temp should be put to zero
		  _gpu_init_Greens_kernel_elements<<<gridsize, blocksize>>>(temp, Nkernel, i, j, FD_cell_size, cst, repetition, qd_P_10, qd_W_10);
			cudaThreadSynchronize();
		}
	}

}

__global__ void _gpu_init_Greens_kernel_elements(tensor *temp, int Nkernel, int co1, int co2, float *FD_cell_size, float cst, int *repetition, float **qd_P_10, float *qd_W_10){
   
    int i = blockIdx.x;
    int j = blockIdx.y;
    int k = threadIdx.x;
		float*** temp_3D = tensor_array3D(temp);
		
		temp_3D[i           ][j           ][k           ] = get_Greens_element(Nkernel, co1, co2,  i,  j,  k, FD_cell_size, cst, repetion, qd_P_10, qd_W_10);
		if (i>0) 
		temp_3D[Nkernel[X]-i][j           ][k           ] = get_Greens_element(Nkernel, co1, co2, -i,  j,  k, FD_cell_size, cst, repetion, qd_P_10, qd_W_10);
		if (j>0) 
		temp_3D[i           ][Nkernel[Y]-j][k           ] = get_Greens_element(Nkernel, co1, co2,  i, -j,  k, FD_cell_size, cst, repetion, qd_P_10, qd_W_10);
		if (k>0) 
		temp_3D[i           ][j           ][Nkernel[Z]-k] = get_Greens_element(Nkernel, co1, co2,  i,  j, -k, FD_cell_size, cst, repetion, qd_P_10, qd_W_10);
		if (i>0 && j>0) 
		temp_3D[Nkernel[X]-i][Nkernel[Y]-j][k           ] = get_Greens_element(Nkernel, co1, co2, -i, -j,  k, FD_cell_size, cst, repetion, qd_P_10, qd_W_10);
		if (i>0 && k>0) 
		temp_3D[Nkernel[X]-i][j           ][Nkernel[Z]-k] = get_Greens_element(Nkernel, co1, co2, -i,  j, -k, FD_cell_size, cst, repetion, qd_P_10, qd_W_10);
		if (j>0 && k>0) 
		temp_3D[i           ][Nkernel[Y]-j][Nkernel[Z]-k] = get_Greens_element(Nkernel, co1, co2,  i, -j, -k, FD_cell_size, cst, repetion, qd_P_10, qd_W_10);
		if (i>0 && j>0 && k>0) 
		temp_3D[Nkernel[X]-i][Nkernel[Y]-j][Nkernel[Z]-k] = get_Greens_element(Nkernel, co1, co2, -i, -j, -k, FD_cell_size, cst, repetion, qd_P_10, qd_W_10);
}

float get_Greens_element(int Nkernel, int co1, int co2, int a, int b, int c, float *FD_cell_size, float *cst, int *repetition, float **qd_P_10, float *qd_W_10){


	float result = 0.0;

	if (co1==0 && co2==0){

		for(int cnta=-repetition[X]; cnta<=repetition[X]; cnta++)
		for(int cntb=-repetition[Y]; cntb<=repetition[Y]; cntb++)
		for(int cntc=-repetition[Z]; cntc<=repetition[Z]; cntc++){

			int i = a + cnta*Nkernel[X]/2;
			int j = b + cntb*Nkernel[Y]/2;
			int k = c + cntc*Nkernel[Z]/2;
			int r2_int = i*i+j*j+k*k;
//			if (r2_int==0.0) continue;    ///include selfpatch?

			if (r2<400.0){
				float x1 = (i + 0.5f) * FD_cell_size[X];
				float x2 = (i - 0.5f) * FD_cell_size[X];
				for (int cnt2=0; cnt2<10; cnt2++){
					float y = j * FD_cell_size[Y] + qd_P_10[Y][cnt2];
					for (int cnt3=0; cnt3<10; cnt3++){
						float z = k * FD_cell_size[Z] + qd_P_10[Z][cnt3];
						result += FD_cell_size[Y] * FD_cell_size[Z] /4.0f * qd_W_10[cnt2] * qd_W_10[cnt3] *
							( x1*fpow(x1*x1+y*y+z*z, -1.5) - x2*fpow(x2*x2+y*y+z*z, -1.5));
					}
				}
			}
			else{
				int r2_int = (i*FD_cell_size[X])*(i*FD_cell_size[X]) + (j*FD_cell_size[Y])*(j*FD_cell_size[Y]) + (k*FD_cell_size[Z])*(k*FD_cell_size[Z]);
				result += FD_cell_size[X] * FD_cell_size[Y] * FD_cell_size[Z] * 
									(1.0f/ fpow(r2,1.5) - 3.0* (i*FD_cell_size[X]) * (i*FD_cell_size[X]) * pow(r2,-2.5));
			}
		}

	return( cst*(result) );
	}
	
		if (co1==0 && co2==1){
		
		}
	
		if (co1==0 && co2==2){
		
		}
		if (co1==1 && co2==1){
		
		}
		if (co1==1 && co2==2){
		
		}
		if (co1==2 && co2==2){
		
		}
	
	
}




/*	
	int cnt;

	gx = NULL;
	gy = NULL;
	gz = NULL;
	FFT_M = NULL;
	FFT_Mx = (float *) fftwf_malloc(N->FFT*sizeof(float));
	FFT_My = (float *) fftwf_malloc(N->FFT*sizeof(float));
	FFT_Mz = (float *) fftwf_malloc(N->FFT*sizeof(float));

	fxx = FFT_Mx;
	fxy = FFT_My;
	fxz = FFT_Mz;
	fyy = FFT_H;
	fyz = (float *) fftwf_malloc(N->FFT*sizeof(float));
	fzz = (float *) fftwf_malloc(N->FFT*sizeof(float));

	memset(fxx, 0, N->FFT*sizeof(float));
	memset(fxy, 0, N->FFT*sizeof(float));
	memset(fxz, 0, N->FFT*sizeof(float));
	memset(fyy, 0, N->FFT*sizeof(float));
	memset(fyz, 0, N->FFT*sizeof(float	{ 
		0.066671344308687999f, 0.149451349150581f  , 0.21908636251598201f, 0.26926671930999602f , 0.29552422471475298f,
		0.29552422471475298f, 0.26926671930999602f , 0.21908636251598201f, 0.149451349150581f, 0.066671344308687999f };

));
	memset(fzz, 0, N->FFT*sizeof(float));

	thread_Wrapper(evaluate_Greens_Functions1);
	printf("\n");

	if (N->n==1){
		fxx[0] -= 6.0*cst->Hexch;
		fyy[0] -= 6.0*cst->Hexch;
		fzz[0] -= 6.0*cst->Hexch;
		fxx[1] += cst->Hexch;
		fyy[1] += cst->Hexch;
		fzz[1] += cst->Hexch;
		fxx[N->FFT_z] += cst->Hexch;
		fyy[N->FFT_z] += cst->Hexch;
		fzz[N->FFT_z] += cst->Hexch;
		fxx[N->FFT_yz] += cst->Hexch;
		fyy[N->FFT_yz] += cst->Hexch;
		fzz[N->FFT_yz] += cst->Hexch;
		fxx[N->FFT_z-3] += cst->Hexch;
		fyy[N->FFT_z-3] += cst->Hexch;
		fzz[N->FFT_z-3] += cst->Hexch;
		fxx[(N->FFT_y-1)*N->FFT_z] += cst->Hexch;
		fyy[(N->FFT_y-1)*N->FFT_z] += cst->Hexch;
		fzz[(N->FFT_y-1)*N->FFT_z] += cst->Hexch;
		fxx[(N->FFT_x-1)*N->FFT_yz] +=cst->Hexch;
		fyy[(N->FFT_x-1)*N->FFT_yz] +=cst->Hexch;
		fzz[(N->FFT_x-1)*N->FFT_yz] +=cst->Hexch;
	}

	fftwf_FW_3d_real_input(N->FFT_dimx, N->FFT_dimy, N->FFT_dimz, fxx, fxx,
    plans_full);
	printf("\tFFT Greens functions     : 16.7 percent\r");
		fflush(stdout);
	fftwf_FW_3d_real_input(N->FFT_dimx, N->FFT_dimy, N->FFT_dimz, fxy, fxy,
    plans_full);
	printf("\tFFT Greens functions     : 33.3 percent\r");
		fflush(stdout);
	fftwf_FW_3d_real_input(N->FFT_dimx, N->FFT_dimy, N->FFT_dimz, fxz, fxz,
    plans_full);
	printf("\tFFT Greens functions     : 50.0 percent\r");
		fflush(stdout);
	fftwf_FW_3d_real_input(N->FFT_dimx, N->FFT_dimy, N->FFT_dimz, fyy, fyy,
    plans_full);
	printf("\tFFT Greens functions     : 66.7 percent\r");
		fflush(stdout);
	fftwf_FW_3d_real_input(N->FFT_dimx, N->FFT_dimy, N->FFT_dimz, fyz, fyz,
    plans_full);
	printf("\tFFT Greens functions     : 83.3 percent\r");
		fflush(stdout);
	fftwf_FW_3d_real_input(N->FFT_dimx, N->FFT_dimy, N->FFT_dimz, fzz, fzz,
    plans_full);
	printf("\tFFT Greens functions     : 100.0 percent\n");
		fflush(stdout);

	gyz = (float *) calloc(N->FFT/2, sizeof(float));
	gzz = (float *) calloc(N->FFT/2, sizeof(float));
	for (cnt=0; cnt<N->FFT/2; cnt++){
		gyz[cnt] = fyz[2*cnt];
		gzz[cnt] = fzz[2*cnt];
	}
	fftwf_free(fyz);
	fftwf_free(fzz);

	gxx = (float *) calloc(N->FFT/2, sizeof(float));
	gxy = (float *) calloc(N->FFT/2, sizeof(float));
	gxz = (float *) calloc(N->FFT/2, sizeof(float));
	gyy = (float *) calloc(N->FFT/2, sizeof(float));

	for (cnt=0; cnt<N->FFT/2; cnt++){
		gxx[cnt] = fxx[2*cnt];
		gxy[cnt] = fxy[2*cnt];
		gxz[cnt] = fxz[2*cnt];
		gyy[cnt] = fyy[2*cnt];
	}*/

// 	FILE *greens = fopen("Greens_functions_80x512x2", "w");
// 	for (cnt=0; cnt<N->FFT/2; cnt++){
// 		fprintf(greens, "%f\t%f\t%f\t%f\t%f\t%f\n", gxx[cnt], gyy[cnt], gzz[cnt], 
// 	    gxy[cnt], gxz[cnt], gyz[cnt]);
// 	}
// 	fclose (greens);

// 	FILE *greens = fopen("Greens_functions_80x1024x2", "w");
// 	for (cnt=0; cnt<N->FFT/2; c]nt++){
// 		fprintf(greens, "%f\t%f\t%f\t%f\t%f\t%f\n", gxx[cnt], gyy[cnt], gzz[cnt], 
// 	    gxy[cnt], gxz[cnt], gyz[cnt]);
// 	}
// 	fclose (greens);

	return;
}


// Gauss quadrature functions ________________________________________________
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

