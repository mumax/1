#include "gpurk4.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define alpha 1.0f

//__________________________________________________________________________________________ step 0

__global__ void _gpu_rk4step_0(float* mx, float* my, float* mz, 
			       float* hx, float* hy, float* hz, 
			       float* kx, float* ky, float* kz, 
			       float dt){
  
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  
 
  // - m cross H
  float _mxHx = -my[i] * hz[i] + hy[i] * mz[i];
  float _mxHy =  mx[i] * hz[i] - hx[i] * mz[i];
  float _mxHz = -mx[i] * hy[i] + hx[i] * my[i];

  // - m cross (m cross H)
  float _mxmxHx =  my[i] * _mxHz - _mxHy * mz[i];
  float _mxmxHy = -mx[i] * _mxHz + _mxHx * mz[i];
  float _mxmxHz =  mx[i] * _mxHy - _mxHx * my[i];

  kx[i] = (_mxHx + _mxmxHx * alpha);
  ky[i] = (_mxHy + _mxmxHy * alpha);
  kz[i] = (_mxHz + _mxmxHz * alpha);
  
  mx[i] += 0.5f * dt * kx[i];
  my[i] += 0.5f * dt * ky[i];
  mz[i] += 0.5f * dt * kz[i];
  
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]); // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
}

//__________________________________________________________________________________________ step 1

__global__ void _gpu_rk4step_1(float*  mx, float*  my, float*  mz, 
			       float*  hx, float*  hy, float*  hz, 
			       float*  kx, float*  ky, float*  kz, 
			       float* m0x, float* m0y, float* m0z,
			       float  dt){
  
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  
 
  // - m cross H
  float _mxHx = -my[i] * hz[i] + hy[i] * mz[i];
  float _mxHy =  mx[i] * hz[i] - hx[i] * mz[i];
  float _mxHz = -mx[i] * hy[i] + hx[i] * my[i];

  // - m cross (m cross H)
  float _mxmxHx =  my[i] * _mxHz - _mxHy * mz[i];
  float _mxmxHy = -mx[i] * _mxHz + _mxHx * mz[i];
  float _mxmxHz =  mx[i] * _mxHy - _mxHx * my[i];

  kx[i] = (_mxHx + _mxmxHx * alpha);
  ky[i] = (_mxHy + _mxmxHy * alpha);
  kz[i] = (_mxHz + _mxmxHz * alpha);
  
  mx[i] = m0x[i] + 0.5f * dt * kx[i];
  my[i] = m0x[i] + 0.5f * dt * ky[i];
  mz[i] = m0x[i] + 0.5f * dt * kz[i];
  
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]); // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
}

//__________________________________________________________________________________________ step 2

__global__ void _gpu_rk4step_2(float*  mx, float*  my, float*  mz, 
			       float*  hx, float*  hy, float*  hz, 
			       float*  kx, float*  ky, float*  kz, 
			       float* m0x, float* m0y, float* m0z,
			       float  dt){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  
 
  // - m cross H
  float _mxHx = -my[i] * hz[i] + hy[i] * mz[i];
  float _mxHy =  mx[i] * hz[i] - hx[i] * mz[i];
  float _mxHz = -mx[i] * hy[i] + hx[i] * my[i];

  // - m cross (m cross H)
  float _mxmxHx =  my[i] * _mxHz - _mxHy * mz[i];
  float _mxmxHy = -mx[i] * _mxHz + _mxHx * mz[i];
  float _mxmxHz =  mx[i] * _mxHy - _mxHx * my[i];

  kx[i] = (_mxHx + _mxmxHx * alpha);
  ky[i] = (_mxHy + _mxmxHy * alpha);
  kz[i] = (_mxHz + _mxmxHz * alpha);
  
  mx[i] = m0x[i] + dt * kx[i];
  my[i] = m0x[i] + dt * ky[i];
  mz[i] = m0x[i] + dt * kz[i];
  
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]); // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
}

//__________________________________________________________________________________________ step 3

__global__ void _gpu_rk4step_3(float*  mx, float*  my, float*  mz, 
			       float*  hx, float*  hy, float*  hz, 
			       float*  kx, float*  ky, float*  kz, 
			       float* m0x, float* m0y, float* m0z,
			       float dt){
  
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  
 
  // - m cross H
  float _mxHx = -my[i] * hz[i] + hy[i] * mz[i];
  float _mxHy =  mx[i] * hz[i] - hx[i] * mz[i];
  float _mxHz = -mx[i] * hy[i] + hx[i] * my[i];

  // - m cross (m cross H)
  float _mxmxHx =  my[i] * _mxHz - _mxHy * mz[i];
  float _mxmxHy = -mx[i] * _mxHz + _mxHx * mz[i];
  float _mxmxHz =  mx[i] * _mxHy - _mxHx * my[i];

  kx[i] = (_mxHx + _mxmxHx * alpha);
  ky[i] = (_mxHy + _mxmxHy * alpha);
  kz[i] = (_mxHz + _mxmxHz * alpha);

  mx[i] = m0x[i];
  my[i] = m0y[i];
  mz[i] = m0z[i];
  
}

//__________________________________________________________________________________________ step 4

__global__ void _gpu_rk4step_4(float* mx,  float* my,  float* mz, 
			       float* k0x, float* k0y, float* k0z,
			       float* k1x, float* k1y, float* k1z,
			       float* k2x, float* k2y, float* k2z,
			       float* k3x, float* k3y, float* k3z,
			       float dt){
  
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
 
  mx[i] += dt * (1.0f/6.0f) * (k0x[i] + 2.0f*(k1x[i] + k2x[i]) + k3x[i]); 
  my[i] += dt * (1.0f/6.0f) * (k0y[i] + 2.0f*(k1y[i] + k2y[i]) + k3y[i]);
  mz[i] += dt * (1.0f/6.0f) * (k0z[i] + 2.0f*(k1z[i] + k2z[i]) + k3z[i]);
  
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]); // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;

}

//__________________________________________________________________________________________ gpurk4_step

void gpurk4_step(gpurk4* solver, float dt){
  
  int threadsPerBlock = 256;
  int blocks = (solver->convplan->len_m_comp) / threadsPerBlock;
  gpu_checkconf_int(blocks, threadsPerBlock);
  
  float* mx = &(solver->m[X*solver->len_m_comp]);
  float* my = &(solver->m[Y*solver->len_m_comp]);
  float* mz = &(solver->m[Z*solver->len_m_comp]);
  
  float* hx = &(solver->h[X*solver->len_m_comp]);
  float* hy = &(solver->h[Y*solver->len_m_comp]);
  float* hz = &(solver->h[Z*solver->len_m_comp]);
  
  float* m0x = &(solver->m0[X*solver->len_m_comp]);
  float* m0y = &(solver->m0[Y*solver->len_m_comp]);
  float* m0z = &(solver->m0[Z*solver->len_m_comp]);
  
  float* k0x = &(solver->k[0][X*solver->len_m_comp]);
  float* k0y = &(solver->k[0][Y*solver->len_m_comp]);
  float* k0z = &(solver->k[0][Z*solver->len_m_comp]);
  
  float* k1x = &(solver->k[1][X*solver->len_m_comp]);
  float* k1y = &(solver->k[1][Y*solver->len_m_comp]);
  float* k1z = &(solver->k[1][Z*solver->len_m_comp]);
  
  float* k2x = &(solver->k[2][X*solver->len_m_comp]);
  float* k2y = &(solver->k[2][Y*solver->len_m_comp]);
  float* k2z = &(solver->k[2][Z*solver->len_m_comp]);

  float* k3x = &(solver->k[3][X*solver->len_m_comp]);
  float* k3y = &(solver->k[3][Y*solver->len_m_comp]);
  float* k3z = &(solver->k[3][Z*solver->len_m_comp]);

  // first backup starting point m0
  memcpy_on_gpu(solver->m, solver->m0, solver->len_m);
  // calc the field
  gpuconv1_exec(solver->convplan, solver->m, solver->h);
  
  _gpu_rk4step_0<<<blocks, threadsPerBlock>>>(mx,my,mz,  hx,hy,hz,  k0x,k0y,k0z,  dt);
  gpu_sync();
  
  gpuconv1_exec(solver->convplan, solver->m, solver->h);

  _gpu_rk4step_1<<<blocks, threadsPerBlock>>>(mx,my,mz,  hx,hy,hz,  k1x,k1y,k1z,  m0x,m0y,m0z,  dt);
  gpu_sync();
  
  gpuconv1_exec(solver->convplan, solver->m, solver->h);
  
  _gpu_rk4step_2<<<blocks, threadsPerBlock>>>(mx,my,mz,  hx,hy,hz,  k2x,k2y,k2z,  m0x,m0y,m0z,  dt);
  gpu_sync();
  
  gpuconv1_exec(solver->convplan, solver->m, solver->h);
  
  _gpu_rk4step_3<<<blocks, threadsPerBlock>>>(mx,my,mz,  hx,hy,hz,  k3x,k3y,k3z,  m0x,m0y,m0z,  dt);
  gpu_sync();
  
  _gpu_rk4step_4<<<blocks, threadsPerBlock>>>(mx,my,mz,  k0x,k0y,k0z,  k1x,k1y,k1z,  k2x,k2y,k2z,  k3x,k3y,k3z,  dt);
  gpu_sync();
}

void gpurk4_checksize_m(gpurk4* sim, tensor* m){
   // m should be a rank 4 tensor with size 3 x N0 x N1 x N2
  assert(m->rank == 4);
  assert(m->size[0] == 3); 
  for(int i=0; i<3; i++){ assert(m->size[i+1] == sim->size[i]); }
}

void gpurk4_loadm(gpurk4* solver, tensor* m){
  gpurk4_checksize_m(solver, m); 
  memcpy_to_gpu(m->list, solver->m, solver->len_m);
}

void gpurk4_storem(gpurk4* solver, tensor* m){
  gpurk4_checksize_m(solver, m); 
  memcpy_from_gpu(solver->m, m->list, solver->len_m);
}

void gpurk4_init_sizes(gpurk4* solver, int N0, int N1, int N2){
  solver->size = (int*)calloc(3, sizeof(int));
  solver->size[0] = N0; solver->size[1] = N1; solver->size[2] = N2;
  solver->N = N0 * N1 * N2;
}

void gpurk4_init_m(gpurk4* solver){
  solver->len_m = 3 * solver->N;
  solver->len_m_comp = solver->N;
  solver->m = new_gpu_array(solver->len_m);
  solver->m0 = new_gpu_array(solver->len_m);
}

void gpurk4_init_h(gpurk4* solver){
  solver->len_h = 3 * solver->N;
  solver->h = new_gpu_array(solver->len_h);
}

void gpurk4_init_k(gpurk4* solver){
  solver->k = (float**)calloc(3, sizeof(float*));
  // painful bugfix: was initialized up to 3 instead of 4.
  // result: error in memcpy_on_gpu, AFTER execution of the offending kernel and acting on legal addresses
  // FOLLOWED by : unspecified launch failure
  for(int i=0; i<4; i++){
    solver->k[i] = new_gpu_array(solver->len_m);
  }
}

gpurk4* new_gpurk4(int N0, int N1, int N2, tensor* kernel){
  gpurk4* solver = (gpurk4*)malloc(sizeof(gpurk4));
  gpurk4_init_sizes(solver, N0, N1, N2);
  gpurk4_init_m(solver);
  gpurk4_init_h(solver);
  gpurk4_init_k(solver);
  solver->convplan = new_gpuconv1(N0, N1, N2, kernel);
  return solver;
}

#ifdef __cplusplus
}
#endif