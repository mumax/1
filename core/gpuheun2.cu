#include "gpuheun2.h"
#include "gputil.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

///@todo these common functions should be moved to a common file like gpusolver ...
__global__ void _gpu_normalize2(float* mx , float* my , float* mz){
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
  float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);     // inverse square root
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
}

void gpuheun2_normalize_m(gpuheun2* solver){
//   int threadsPerBlock = 512;
//   int blocks = (solver->convplan->len_mComp) / threadsPerBlock;
//   gpu_checkconf_int(blocks, threadsPerBlock);
  
  int gridSize = -1, blockSize = -1;
  make1dconf(solver->mComp[X]->len, &gridSize, &blockSize);
  
  timer_start("normalize_m");
  _gpu_normalize2<<<gridSize, blockSize>>>(solver->mComp[X]->list, solver->mComp[Y]->list, solver->mComp[Z]->list);
  cudaThreadSynchronize();
  timer_stop("normalize_m");
}


__global__ void _gpu_heun2step0(float* mx , float* my , float* mz ,
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


__global__ void _gpu_heun2step1(float* mx , float* my , float* mz ,
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


void gpuheun2_step_old(gpuheun2* solver, float dt, float alpha){
  
  int gridSize = -1, blockSize = -1;
  make1dconf(solver->mComp[X]->len, &gridSize, &blockSize);
  
//   tensor** m = solver->mComp;
//   tensor** m0 = solver->m0_comp;
//   tensor** h = solver->hComp;
//   tensor** t0 = solver->torque0_comp;
//   float* hExt = solver->hExt;
  
  //memcpy_gpu_to_gpu(solver->m, solver->m0, solver->len_m);
  tensor_copy_gpu_to_gpu(solver->m, solver->m0);

	printf("gpuheun2: Been here1\n");
  gpuconv2_exec(solver->convplan, solver->m, solver->h);
  
  timer_start("gpuheun_step");
  _gpu_heun2step0<<<gridSize, blockSize>>>(solver->      mComp[X]->list,  solver->      mComp[Y]->list,  solver->      mComp[Z]->list,
                                           solver->      hComp[X]->list,  solver->      hComp[Y]->list,  solver->      hComp[Z]->list,
                                           solver->torque0Comp[X]->list,  solver->torque0Comp[Y]->list,  solver->torque0Comp[Z]->list,
                                           solver->       hExt[X],        solver->       hExt[Y],        solver->       hExt[Z],
                                           dt, alpha);
  cudaThreadSynchronize();
  timer_stop("gpuheun_step");
  gpuheun2_normalize_m(solver);
   
	printf("gpuheun2: Been here2\n");
  gpuconv2_exec(solver->convplan, solver->m, solver->h);
  
  timer_start("gpuheun_step");
  _gpu_heun2step1<<<gridSize, blockSize>>>(solver->      mComp[X]->list,  solver->      mComp[Y]->list,  solver->      mComp[Z]->list,
                                           solver->      hComp[X]->list,  solver->      hComp[Y]->list,  solver->      hComp[Z]->list,
                                           solver->torque0Comp[X]->list,  solver->torque0Comp[Y]->list,  solver->torque0Comp[Z]->list,
                                           solver->     m0Comp[X]->list,  solver->     m0Comp[Y]->list,  solver->     m0Comp[Z]->list,
                                           solver->       hExt[X],             solver->  hExt[Y],             solver->  hExt[Z],
                                           0.5f*dt, alpha);
  //(m[X],m[Y],m[Z],  h[X],h[Y],h[Z],  t0[X],t0[Y],t0[Z], m0[X], m0[Y], m0[Z], hExt[X],hExt[Y],hExt[Z], 0.5f*dt, alpha);
  
  cudaThreadSynchronize();
  timer_stop("gpuheun_step");
  gpuheun2_normalize_m(solver);
}

void gpuheun2_step(gpuheun2* solver){
  gpuheun2_step_old(solver, solver->params->maxDt, solver->params->alpha); ///@todo get rid of _old
}

// void gpuheun2_checksize_m(gpuheun2* sim, tensor* m){
//    // m should be a rank 4 tensor with size 3 x N0 x N1 x N2
//   assert(m->rank == 4);
//   assert(m->size[0] == 3); 
//   for(int i=0; i<3; i++){ assert(m->size[i+1] == sim->size[i]); }
// }

void gpuheun2_loadm(gpuheun2* heun, tensor* m){
  assert(m->rank == 4);
  assert(m->size[0] == 3);
  tensor_copy_to_gpu(m, heun->m);       //checks the sizes too
}

void gpuheun2_storem(gpuheun2* heun, tensor* m){
  assert(m->rank == 4);
  assert(m->size[0] == 3);
  tensor_copy_from_gpu(heun->m, m);
}

void gpuheun2_storeh(gpuheun2* heun, tensor* h){
   assert(h->rank == 4);
   assert(h->size[0] == 3);
   tensor_copy_from_gpu(heun->h, h);
}

// void gpuheun2_init_m(gpuheun2* heun){
// 
//   heun->m = new_gputensor(heun->len_m);
//   heun->m0 = new_gpu_array(heun->len_m);
//   heun->mComp = (float**)calloc(3, sizeof(float*));
//   heun->m0_comp = (float**)calloc(3, sizeof(float*));
//   for(int i=0; i<3; i++){
//     heun->mComp [i] = &(heun->m [i * heun->len_mComp]);
//     heun->m0_comp[i] = &(heun->m0[i * heun->len_mComp]);
//   }
// }
// 
// void gpuheun2_init_h(gpuheun2* heun){
//   heun->h = new_gpu_array(heun->len_m);
//   heun->torque0 = new_gpu_array(heun->len_m);
//   heun->hComp = (float**)calloc(3, sizeof(float*));
//   heun->torque0_comp = (float**)calloc(3, sizeof(float*));
//   for(int i=0; i<3; i++){
//     heun->hComp[i] = &(heun->h[i * heun->len_mComp]);
//     heun->torque0_comp[i] = &(heun->torque0[i * heun->len_mComp]);
//   }
// }
// 
// void gpuheun2_init_hExt(gpuheun2* solver, float* hExt){
//   solver->hExt = (float*)calloc(3, sizeof(float));
//   for(int i=0; i<3; i++){
//     solver->hExt[i] = hExt[i];
//   }
// }

gpuheun2* new_gpuheun2_param(param* p, tensor* kernel){
  
  gpuheun2* heun = (gpuheun2*)malloc(sizeof(gpuheun2));

  heun->params = p;
  
  int* size4D = tensor_size4D(p->size);
  assert(size4D[0] == 3);
  int* kernelSize = p->demagKernelSize;

  //fprintf(stderr, "new_gpuheun2([%d x %d x %d],[%d x %d x %d],[%g, %g, %g])\n", p->size[X], p->size[Y], p->size[Z], kernelSize[X], kernelSize[Y], kernelSize[Z], p->hExt[X], p->hExt[Y], p->hExt[Z]);

  heun->m       = new_gputensor(4, size4D);
  heun->m0      = new_gputensor(4, size4D);
  heun->h       = new_gputensor(4, size4D);
  heun->torque0 = new_gputensor(4, size4D);

  for(int i=0; i<3; i++){
    heun->      mComp[i] = tensor_component(heun->m,       i);
    heun->     m0Comp[i] = tensor_component(heun->m0,      i);
    heun->      hComp[i] = tensor_component(heun->h,       i);
    heun->torque0Comp[i] = tensor_component(heun->torque0, i);
  }

  heun->convplan = new_gpuconv2(p->size, p->demagKernelSize);
  gpuconv2_loadkernel5DSymm(heun->convplan, kernel);

  heun->hExt = p->hExt; 
  fprintf(stderr, "hExt: %f %f %f\n", heun->hExt[X], heun->hExt[Y], heun->hExt[Z]);


  fprintf(stderr, "new_gpuheun2(): OK\n");
  return heun;
}


gpuheun2* new_gpuheun2(int* size, tensor* kernel, float* hExt){
  
  gpuheun2* heun = (gpuheun2*)malloc(sizeof(gpuheun2));

  heun->params = NULL;
  
  int* size4D = tensor_size4D(size);
  assert(size4D[0] == 3);
  int* kernelSize = (int*)safe_calloc(3, sizeof(int));
  kernelSize[X] = kernel->size[2+X];
  kernelSize[Y] = kernel->size[2+Y];
  kernelSize[Z] = kernel->size[2+Z];

  fprintf(stderr, "new_gpuheun2([%d x %d x %d],[%d x %d x %d],[%g, %g, %g])\n", size[X], size[Y], size[Z], kernelSize[X], kernelSize[Y], kernelSize[Z], hExt[X], hExt[Y], hExt[Z]);
  
  heun->m       = new_gputensor(4, size4D);
  heun->m0      = new_gputensor(4, size4D);
  heun->h       = new_gputensor(4, size4D);
  heun->torque0 = new_gputensor(4, size4D);
  
  for(int i=0; i<3; i++){
    heun->      mComp[i] = tensor_component(heun->m,       i);
    heun->     m0Comp[i] = tensor_component(heun->m0,      i);
    heun->      hComp[i] = tensor_component(heun->h,       i);
    heun->torque0Comp[i] = tensor_component(heun->torque0, i);
  }
  
  heun->convplan = new_gpuconv2(size, kernelSize);
  gpuconv2_loadkernel5DSymm(heun->convplan, kernel);
  
  heun->hExt = (float*)calloc(3, sizeof(float));
  for(int i=0; i<3; i++){
    heun->hExt[i] = hExt[i];
  }
  fprintf(stderr, "hExt: %f %f %f\n", heun->hExt[X], heun->hExt[Y], heun->hExt[Z]);

  
  fprintf(stderr, "new_gpuheun2(): OK\n");
  return heun;
}

#ifdef __cplusplus
}
#endif