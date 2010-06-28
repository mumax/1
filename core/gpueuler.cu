#include "gpueuler.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define alpha 1.0f


__global__ void _gpu_euler_stage(float* mx, float* my, float* mz,
                                 float*tx, float* ty, float* tz){

  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);

  mx[i] += tx[i];
  my[i] += ty[i];
  mz[i] += tz[i];
  
}

void gpueuler_stage(float* m, float* torque, int N){

  int gridSize = -1, blockSize = -1;
  make1dconf(N, &gridSize, &blockSize);

  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  float* tqx = &(torque[0*N]);
  float* tqy = &(torque[1*N]);
  float* tqz = &(torque[2*N]);


  timer_start("euler_stage");
  _gpu_euler_stage<<<gridSize, blockSize>>>(mx, my, mz, tqx, tqy, tqz);
  cudaThreadSynchronize();
  timer_stop("euler_stage");
  
}








__global__ void _gpu_eulerstep(float* mx, float* my, float* mz, float* hx, float* hy, float* hz, float dt){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
 
  // - m cross H
  float _mxHx = -my[i] * hz[i] + hy[i] * mz[i];
  float _mxHy =  mx[i] * hz[i] - hx[i] * mz[i];
  float _mxHz = -mx[i] * hy[i] + hx[i] * my[i];

  // - m cross (m cross H)
  float _mxmxHx =  my[i] * _mxHz - _mxHy * mz[i];
  float _mxmxHy = -mx[i] * _mxHz + _mxHx * mz[i];
  float _mxmxHz =  mx[i] * _mxHy - _mxHx * my[i];

  float torquex = (_mxHx + _mxmxHx * alpha);
  float torquey = (_mxHy + _mxmxHy * alpha);
  float torquez = (_mxHz + _mxmxHz * alpha);
  
  mx[i] += torquex * dt;
  my[i] += torquey * dt;
  mz[i] += torquez * dt;
  
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]); // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
}

void gpueuler_step(gpueuler* solver, tensor* m, tensor* h, double* dt){
//   int threadsPerBlock = 512;
//   gpuconv1_exec(solver->convplan, solver->m, solver->h);
//   int blocks = (solver->convplan->len_m_comp) / threadsPerBlock;
//   gpu_checkconf_int(blocks, threadsPerBlock);
//   timer_start("gpueuler_step");
//   _gpu_eulerstep<<<blocks, threadsPerBlock>>>(solver->convplan->m_comp[0], solver->convplan->m_comp[1], solver->convplan->m_comp[2], solver->convplan->h_comp[0], solver->convplan->h_comp[1], solver->convplan->h_comp[2], dt);
//   cudaThreadSynchronize();
//   timer_stop("gpueuler_step");
}

void gpueuler_checksize_m(gpueuler* sim, tensor* m){
   // m should be a rank 4 tensor with size 3 x N0 x N1 x N2
  assert(m->rank == 4);
  assert(m->size[0] == 3); 
  for(int i=0; i<3; i++){ assert(m->size[i+1] == sim->size[i]); }
}

void gpueuler_loadm(gpueuler* euler, tensor* m){
  gpueuler_checksize_m(euler, m); 
  memcpy_to_gpu(m->list, euler->m, euler->len_m);
}

void gpueuler_storem(gpueuler* euler, tensor* m){
  gpueuler_checksize_m(euler, m); 
  memcpy_from_gpu(euler->m, m->list, euler->len_m);
}

void gpueuler_init_sizes(gpueuler* euler, int N0, int N1, int N2){
  euler->size = (int*)calloc(3, sizeof(int));
  euler->size[0] = N0; euler->size[1] = N1; euler->size[2] = N2;
  euler->N = N0 * N1 * N2;
}

void gpueuler_init_m(gpueuler* euler){
  euler->len_m = 3 * euler->N;
  euler->m = new_gpu_array(euler->len_m);
}

void gpueuler_init_h(gpueuler* euler){
  euler->len_h = 3 * euler->N;
  euler->h = new_gpu_array(euler->len_h);
}

gpueuler* new_gpueuler(param* p){
//   gpueuler* euler = (gpueuler*)malloc(sizeof(gpueuler));
//   gpueuler_init_sizes(euler, N0, N1, N2);
//   gpueuler_init_m(euler);
//   gpueuler_init_h(euler);
//   euler->convplan = new_gpuconv1(N0, N1, N2, kernel);
//   return euler;
}

#ifdef __cplusplus
}
#endif