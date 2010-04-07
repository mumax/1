#include "gpurk4.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void _gpu_rk4step_1(float* mx, float* my, float* mz, float* hx, float* hy, float* hz, float* kx, float* ky, float* kz){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  float alpha = 0.2;
 
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
}

__global__ void _gpu_rk4step_2(float* mx,  float* my,  float* mz, 
			       float* m0x, float* m0y, float* m0z,
			       float* kx, float* ky, float* kz, 
			       float dt){
  
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
 
  mx[i] = m0x[i] + kx[i] * dt;
  my[i] = m0y[i] + ky[i] * dt;
  mz[i] = m0z[i] + kz[i] * dt;
  
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]); // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;

}

void gpurk4_step(gpurk4* solver, float dt){
  int threadsPerBlock = 512;
  int blocks = (solver->convplan->len_m_comp) / threadsPerBlock;
  gpu_checkconf_int(blocks, threadsPerBlock);
  
  gpuconv1_exec(solver->convplan, solver->m, solver->h);
  memcpy_gpu_to_gpu(solver->m, solver->m0, solver->len_m);
  
  float* k0x = &(solver->k[0][X*solver->len_m_comp]);
  float* k0y = &(solver->k[0][Y*solver->len_m_comp]);
  float* k0z = &(solver->k[0][Z*solver->len_m_comp]);
  
  _gpu_rk4step_1<<<blocks, threadsPerBlock>>>(solver->convplan->m_comp[0], solver->convplan->m_comp[1], solver->convplan->m_comp[2],
					      solver->convplan->h_comp[0], solver->convplan->h_comp[1], solver->convplan->h_comp[2], 
					      k0x, k0y, k0z);
					      
  float* m0x = &(solver->m0[X*solver->len_m_comp]);
  float* m0y = &(solver->m0[Y*solver->len_m_comp]);
  float* m0z = &(solver->m0[Z*solver->len_m_comp]);
  
  _gpu_rk4step_2<<<blocks, threadsPerBlock>>>(solver->convplan->m_comp[X], solver->convplan->m_comp[Y], solver->convplan->m_comp[Z],
					      m0x, m0y, m0z,
					      k0x, k0y, k0z,
					      dt);
					      
  
  cudaThreadSynchronize();
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
}

void gpurk4_init_h(gpurk4* solver){
  solver->len_h = 3 * solver->N;
  solver->h = new_gpu_array(solver->len_h);
}

void gpurk4_init_k(gpurk4* solver){
  solver->k = (float**)calloc(3, sizeof(float*));
  for(int i=0; i<3; i++){
    solver->k[i] = new_gpu_array(solver->len_m);
  }
  solver->m0 = new_gpu_array(solver->len_m);
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