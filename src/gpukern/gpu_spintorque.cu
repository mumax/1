#include "gpu_spintorque.h"
#include "gpu_safe.h"
#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif





/// 2D, plane per plane, i=plane index
__global__ void _gpu_spintorque_deltaM(float* mx, float* my, float* mz,
                            float* hx, float* hy, float* hz,
                            float alpha, float beta,
                            float ux, float uy, float uz,
                            float dt_gilb,
                            int N0, int N1, int N2, int i){

  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;
  int I = i*N1*N2 + j*N2 + k;
  
  if (j < N1 && k < N2){

    // (1) calculate the directional derivative of (mx, my mz) along (ux,uy,uz).
    // Result is (diffmx, diffmy, diffmz)
    // (ux, uy, uz) is 0.5 * U_spintorque / cellsize(x, y, z)
    
    //float m0x = mx[i*N1*N2 + j*N2 + k];
    float mx1 = 0.f, mx2 = 0.f, my1 = 0.f, my2 = 0.f, mz1 = 0.f, mz2 = 0.f;

    // derivative in X direction
    if (i-1 >= 0){
      int idx = (i-1)*N1*N2 + j*N2 + k;
      mx1 = mx[idx];
      my1 = my[idx];
      mz1 = mz[idx];
    } else {
     //
    }
    if (i+1 < N0){
      int idx = (i+1)*N1*N2 + j*N2 + k;
      mx2 = mx[idx];
      my2 = my[idx];
      mz2 = mz[idx];
    } else {
      //
    } 
    float diffmx = ux * (mx2 - mx1);
    float diffmy = uy * (my2 - my1);
    float diffmz = uz * (mz2 - mz1);


    // derivative in Y direction
    if (j-1 >= 0){
      int idx = (i)*N1*N2 + (j-1)*N2 + k;
      mx1 = mx[idx];
      my1 = my[idx];
      mz1 = mz[idx];
    } else {
      //
    } 
    if (j+1 < N1){
      int idx = (i)*N1*N2 + (j+1)*N2 + k;
      mx2 = mx[idx];
      my2 = my[idx];
      mz2 = mz[idx];
    } else {
      //
    } 
    diffmx += ux * (mx2 - mx1);
    diffmy += uy * (my2 - my1);
    diffmz += uz * (mz2 - mz1);


    // derivative in Z direction
    if (k-1 >= 0){
      int idx = (i)*N1*N2 + (j)*N2 + (k-1);
      mx1 = mx[idx];
      my1 = my[idx];
      mz1 = mz[idx];
    } else {
      //
    } 
    if (k+1 < N2){
      int idx = (i)*N1*N2 + (j)*N2 + (k+1);
      mx2 = mx[idx];
      my2 = my[idx];
      mz2 = mz[idx];
    } else {
      //
    } 
    diffmx += ux * (mx2 - mx1);
    diffmy += uy * (my2 - my1);
    diffmz += uz * (mz2 - mz1);


    //(2) torque terms
    // - m cross H
    float Hx = hx[I];
    float Hy = hy[I];
    float Hz = hz[I];
    
    float Mx = mx[I], My = my[I], Mz = mz[I];
    
    float _mxHx = -My * Hz + Hy * Mz;
    float _mxHy =  Mx * Hz - Hx * Mz;
    float _mxHz = -Mx * Hy + Hx * My;

    // - m cross (m cross H)
    float _mxmxHx =  My * _mxHz - _mxHy * Mz;
    float _mxmxHy = -Mx * _mxHz + _mxHx * Mz;
    float _mxmxHz =  Mx * _mxHy - _mxHx * My;
    
    hx[I] = dt_gilb * (_mxHx + _mxmxHx * alpha + diffmx);
    hy[I] = dt_gilb * (_mxHy + _mxmxHy * alpha + diffmy);
    hz[I] = dt_gilb * (_mxHz + _mxmxHz * alpha + diffmz);
  }
  
}


#define BLOCKSIZE 16

void gpu_spintorque_deltaM(float* m, float* h, float alpha, float beta, float* u, float dt_gilb, int N0,  int N1, int N2){

  dim3 gridsize(divUp(N1, BLOCKSIZE), divUp(N2, BLOCKSIZE));
  dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
  int N = N0 * N1 * N2;
  
  for(int i=0; i<N0; i++){
    _gpu_spintorque_deltaM<<<gridsize, blocksize>>>(&m[0*N], &m[1*N], &m[2*N], &h[0*N], &h[1*N], &h[2*N], alpha, beta, u[0], u[1], u[2], dt_gilb, N0, N1, N2, i);
  }
  gpu_sync();
}





#define BLOCKSIZE 16

///@todo this is a slowish test implementation, use shared memory to avoid multiple reads (at least per plane)
__global__ void _gpu_directional_diff2D(float ux, float uy, float uz, float* in, float* out, int N0, int N1, int N2, int i){

//int i = i;
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int k = blockIdx.y * blockDim.y + threadIdx.y;

  if (j < N1 && k < N2){

    float m0 = in[i*N1*N2 + j*N2 + k];
    float mx1, mx2;
    
    if (i-1 >= 0){ mx1 = in[(i-1)*N1*N2 + j*N2 + k]; } else { mx1 = m0; } // not 100% accurate
    if (i+1 < N0){ mx2 = in[(i+1)*N1*N2 + j*N2 + k]; } else { mx2 = m0; } // not 100% accurate
    float answer =  ux * (mx2 - mx1);

    if (j-1 >= 0){ mx1 = in[(i)*N1*N2 + (j-1)*N2 + k]; } else { mx1 = m0; } // not 100% accurate
    if (j+1 < N1){ mx2 = in[(i)*N1*N2 + (j+1)*N2 + k]; } else { mx2 = m0; } // not 100% accurate
    answer += uy * (mx2 - mx1);

    if (k-1 >= 0){ mx1 = in[(i)*N1*N2 + (j)*N2 + (k-1)]; } else { mx1 = m0; } // not 100% accurate
    if (k+1 < N2){ mx2 = in[(i)*N1*N2 + (j)*N2 + (k+1)]; } else { mx2 = m0; } // not 100% accurate
    answer += uz * (mx2 - mx1);

    out[i*N1*N2 + j*N2 + k] = answer;
  }
}


void gpu_directional_diff2D_async(float ux, float uy, float uz, float *input, float *output, int N0, int N1, int N2, int i){
    dim3 gridsize(divUp(N1, BLOCKSIZE), divUp(N2, BLOCKSIZE));
    dim3 blocksize(BLOCKSIZE, BLOCKSIZE, 1);
    _gpu_directional_diff2D<<<gridsize, blocksize>>>(ux, uy, uz, input, output, N0, N1, N2, i);
}

void gpu_directionial_diff(float ux, float uy, float uz, float* in, float* out, int N0, int N1, int N2){
  for(int i=0; i<N0; i++){
    gpu_directional_diff2D_async(ux, uy, uz, &in[i*N1*N2], &out[i*N1*N2], N0, N1, N2, i);
  }
  gpu_sync();
}


#ifdef __cplusplus
}
#endif
