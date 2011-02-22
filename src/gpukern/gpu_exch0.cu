#include "gpu_exch.h"
#include "gpu_safe.h"
#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @todo wrap
/// 2D, plane per plane, i=plane index
__global__ void _gpu_add_exch6(float* mx, float* my, float* mz,
                               float* hx, float* hy, float* hz,
                               int N0, int N1, int N2,
                               int wrap0, int wrap1, int wrap2,
							   float fac0, float fac1, float fac2, 
							   int i){

  //  i is passed
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int I = i*N1*N2 + j*N2 + k; // linear array index
  
  if (j < N1 && k < N2){

    // Local H initiated with central cell contribution. 
	float Hx = - 6.0f * mx[I];
	float Hy = - 6.0f * my[I];
	float Hz = - 6.0f * mz[I];

	// Now add Neighbors.

    // neighbors in X direction
	int idx;
    if (i-1 >= 0){
      idx = (i-1)*N1*N2 + j*N2 + k;
    } else {
      idx = I;
    }
    Hx += mx[idx]; Hy += my[idx]; Hz += mz[idx];

 	if (i+1 < N0){
      idx = (i+1)*N1*N2 + j*N2 + k;
    } else {
      idx = I;
    } 
    Hx += mx[idx]; Hy += my[idx]; Hz += mz[idx];

    // neighbors in Y direction
    if (j-1 >= 0){
      idx = (i)*N1*N2 + (j-1)*N2 + k;
    } else {
      idx = I;
    }
    Hx += mx[idx]; Hy += my[idx]; Hz += mz[idx];

 	if (j+1 < N1){
      idx = (i)*N1*N2 + (j+1)*N2 + k;
    } else {
      idx = I;
    } 
    Hx += mx[idx]; Hy += my[idx]; Hz += mz[idx];

    // neighbors in Z direction
    if (k-1 >= 0){
      idx = (i)*N1*N2 + (j)*N2 + (k-1);
    } else {
      idx = I;
    }
    Hx += mx[idx]; Hy += my[idx]; Hz += mz[idx];

 	if (k+1 < N2){
      idx =  (i)*N1*N2 + (j)*N2 + (k+1);
    } else {
      idx = I;
    } 
    Hx += mx[idx]; Hy += my[idx]; Hz += mz[idx];
   
	// Write back to global memory, add to h 
    hx[I] += fac0 * Hx;
    hy[I] += fac1 * Hy;
    hz[I] += fac2 * Hz;
  }
  
}


#define BLOCKSIZE 16
void gpu_add_exch(float* m, float* h, int N0, int N1, int N2, int wrap0, int wrap1, int wrap2, float cellsize0, float cellsize1, float cellsize2, int type){
  assert(type == 6);
  dim3 gridsize(divUp(N1, BLOCKSIZE), divUp(N2, BLOCKSIZE));
  dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
  int N = N0 * N1 * N2;

  float fac0 = 1.0f/(cellsize0 * cellsize0);
  float fac1 = 1.0f/(cellsize1 * cellsize1);
  float fac2 = 1.0f/(cellsize2 * cellsize2);

  for(int i=0; i<N0; i++){
    _gpu_add_exch6<<<gridsize, blocksize>>>(&m[0*N], &m[1*N], &m[2*N], &h[0*N], &h[1*N], &h[2*N], N0, N1, N2, wrap0, wrap1, wrap2, fac0, fac1, fac2, i);
  }
  gpu_sync();
}



#ifdef __cplusplus
}
#endif
