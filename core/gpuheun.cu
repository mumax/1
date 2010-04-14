#include "gpuheun.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void _gpu_normalize(float* mx , float* my , float* mz){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]); // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
}

void gpuheun_normalize_m(gpuheun* solver){
  int threadsPerBlock = 512;
  int blocks = (solver->convplan->len_m_comp) / threadsPerBlock;
  gpu_checkconf_int(blocks, threadsPerBlock);
  
  timer_start("normalize_m");
  _gpu_normalize<<<blocks, threadsPerBlock>>>(solver->m_comp[X], solver->m_comp[Y], solver->m_comp[Z]);
  cudaThreadSynchronize();
  timer_stop("normalize_m");
}

__global__ void _gpu_heunstep0(float* mx , float* my , float* mz ,
			       float* hx , float* hy , float* hz ,
			       float* t0x, float* t0y, float* t0z,
			       float hExtx, float hExty, float hExtz,
			       float gilbert_dt, float alpha){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
 
  // H total
  float Hx = hx[i] + hExtx;
  float Hy = hy[i] + hExty;
  float Hz = hz[i] + hExtz;
  
  // - m cross H
  float _mxHx = -my[i] * Hz + Hy * mz[i];
  float _mxHy =  mx[i] * Hz - Hx * mz[i];
  float _mxHz = -mx[i] * Hy + Hx * my[i];

  // - m cross (m cross H)
  float _mxmxHx =  my[i] * _mxHz - _mxHy * mz[i];
  float _mxmxHy = -mx[i] * _mxHz + _mxHx * mz[i];
  float _mxmxHz =  mx[i] * _mxHy - _mxHx * my[i];

  t0x[i] = (_mxHx + _mxmxHx * alpha);
  t0y[i] = (_mxHy + _mxmxHy * alpha);
  t0z[i] = (_mxHz + _mxmxHz * alpha);
  
  mx[i] += gilbert_dt * t0x[i];
  my[i] += gilbert_dt * t0y[i];
  mz[i] += gilbert_dt * t0z[i];
}

__global__ void _gpu_heunstep1(float* mx , float* my , float* mz ,
			       float* hx , float* hy , float* hz ,
			       float* t0x, float* t0y, float* t0z,
			       float* m0x, float* m0y, float* m0z,
			       float hExtx, float hExty, float hExtz,
			       float half_gilbert_dt, float alpha){
  
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
 
  // H total
  float Hx = hx[i] + hExtx;
  float Hy = hy[i] + hExty;
  float Hz = hz[i] + hExtz;
  
  // - m cross H
  float _mxHx = -my[i] * Hz + Hy * mz[i];
  float _mxHy =  mx[i] * Hz - Hx * mz[i];
  float _mxHz = -mx[i] * Hy + Hx * my[i];

  // - m cross (m cross H)
  float _mxmxHx =  my[i] * _mxHz - _mxHy * mz[i];
  float _mxmxHy = -mx[i] * _mxHz + _mxHx * mz[i];
  float _mxmxHz =  mx[i] * _mxHy - _mxHx * my[i];

  float torquex = (_mxHx + _mxmxHx * alpha);
  float torquey = (_mxHy + _mxmxHy * alpha);
  float torquez = (_mxHz + _mxmxHz * alpha);
  
  mx[i] = m0x[i] + half_gilbert_dt * (t0x[i] + torquex);
  my[i] = m0y[i] + half_gilbert_dt * (t0y[i] + torquey);
  mz[i] = m0z[i] + half_gilbert_dt * (t0z[i] + torquez);
}

/** @todo gilbert factor */
void gpuheun_step(gpuheun* solver, float dt, float alpha){
  int threadsPerBlock = 512;
  int blocks = (solver->convplan->len_m_comp) / threadsPerBlock;
  gpu_checkconf_int(blocks, threadsPerBlock);
  
  float** m = solver->m_comp;
  float** m0 = solver->m0_comp;
  float** h = solver->h_comp;
  float** t0 = solver->torque0_comp;
  float* hExt = solver->hExt;
  
  memcpy_gpu_to_gpu(solver->m, solver->m0, solver->len_m);
  
  gpuconv1_exec(solver->convplan, solver->m, solver->h);
  
  timer_start("gpuheun_step");
  _gpu_heunstep0<<<blocks, threadsPerBlock>>>(m[X],m[Y],m[Z],  h[X],h[Y],h[Z],  t0[X],t0[Y],t0[Z], hExt[X],hExt[Y],hExt[Z], dt, alpha);
  cudaThreadSynchronize();
  timer_stop("gpuheun_step");
  gpuheun_normalize_m(solver);
   
  gpuconv1_exec(solver->convplan, solver->m, solver->h);
  
  timer_start("gpuheun_step");
  _gpu_heunstep1<<<blocks, threadsPerBlock>>>(m[X],m[Y],m[Z],  h[X],h[Y],h[Z],  t0[X],t0[Y],t0[Z], m0[X], m0[Y], m0[Z], hExt[X],hExt[Y],hExt[Z], 0.5f*dt, alpha);
  cudaThreadSynchronize();
  timer_stop("gpuheun_step");
  gpuheun_normalize_m(solver);
}

void gpuheun_checksize_m(gpuheun* sim, tensor* m){
   // m should be a rank 4 tensor with size 3 x N0 x N1 x N2
  assert(m->rank == 4);
  assert(m->size[0] == 3); 
  for(int i=0; i<3; i++){ assert(m->size[i+1] == sim->size[i]); }
}

void gpuheun_loadm(gpuheun* heun, tensor* m){
  gpuheun_checksize_m(heun, m); 
  memcpy_to_gpu(m->list, heun->m, heun->len_m);
}

void gpuheun_storem(gpuheun* heun, tensor* m){
  gpuheun_checksize_m(heun, m); 
  memcpy_from_gpu(heun->m, m->list, heun->len_m);
}

void gpuheun_storeh(gpuheun* heun, tensor* h){
  gpuheun_checksize_m(heun, h); 
  memcpy_from_gpu(heun->h, h->list, heun->len_m);
}

void gpuheun_init_sizes(gpuheun* heun, int N0, int N1, int N2){
  heun->size = (int*)calloc(3, sizeof(int));
  heun->size[0] = N0; heun->size[1] = N1; heun->size[2] = N2;
  heun->N = N0 * N1 * N2;
}


void gpuheun_init_m(gpuheun* heun){
  heun->len_m = 3 * heun->N;
  heun->len_m_comp = heun->N;
  heun->m = new_gpu_array(heun->len_m);
  heun->m0 = new_gpu_array(heun->len_m);
  heun->m_comp = (float**)calloc(3, sizeof(float*));
  heun->m0_comp = (float**)calloc(3, sizeof(float*));
  for(int i=0; i<3; i++){
    heun->m_comp [i] = &(heun->m [i * heun->len_m_comp]);
    heun->m0_comp[i] = &(heun->m0[i * heun->len_m_comp]);
  }
}

void gpuheun_init_h(gpuheun* heun){
  heun->h = new_gpu_array(heun->len_m);
  heun->torque0 = new_gpu_array(heun->len_m);
  heun->h_comp = (float**)calloc(3, sizeof(float*));
  heun->torque0_comp = (float**)calloc(3, sizeof(float*));
  for(int i=0; i<3; i++){
    heun->h_comp[i] = &(heun->h[i * heun->len_m_comp]);
    heun->torque0_comp[i] = &(heun->torque0[i * heun->len_m_comp]);
  }
}

void gpuheun_init_hExt(gpuheun* solver, float* hExt){
  solver->hExt = (float*)calloc(3, sizeof(float));
  for(int i=0; i<3; i++){
    solver->hExt[i] = hExt[i];
  }
}

gpuheun* new_gpuheun(int N0, int N1, int N2, tensor* kernel, float* hExt){
  gpuheun* heun = (gpuheun*)malloc(sizeof(gpuheun));
  gpuheun_init_sizes(heun, N0, N1, N2);
  gpuheun_init_m(heun);
  gpuheun_init_h(heun);
  heun->convplan = new_gpuconv1(N0, N1, N2, kernel);
  gpuheun_init_hExt(heun, hExt);
  fprintf(stderr, "hExt: %f %f %f\n", heun->hExt[X], heun->hExt[Y], heun->hExt[Z]);
  return heun;
}

#ifdef __cplusplus
}
#endif