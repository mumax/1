#include "gpu_heun.h"
#include "gpukern.h"
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


__global__ void _gpu_heunstage0(float* mx , float* my , float* mz ,
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


__global__ void _gpu_heunstage1(float* mx , float* my , float* mz ,
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


void gpuheun_stage0(gpuheun* solver, tensor* m, tensor* h, double* totalTime){

  dim3 gridSize, blockSize;
  make1dconf(solver->mComp[X]->len, &gridSize, &blockSize); ///@todo cache in heun struct

  timer_start("gpuheun_step");{

    tensor_copy_on_gpu(solver->m, solver->m0);
    _gpu_heunstage0<<<gridSize, blockSize>>>(solver->       mComp[X]->list, solver->       mComp[Y]->list,  solver->       mComp[Z]->list,
                                             solver->       hComp[X]->list, solver->       hComp[Y]->list,  solver->       hComp[Z]->list,
                                             solver-> torque0Comp[X]->list, solver-> torque0Comp[Y]->list,  solver-> torque0Comp[Z]->list,
                                             solver->params->hExt[X],       solver->params->hExt[Y],        solver->params->hExt[Z],
                                             1.0f * solver->params->maxDt, solver->params->alpha);
    gpu_sync();

  }timer_stop("gpuheun_step");
}

void gpuheun_stage1(gpuheun* solver, tensor* m, tensor* h, double* totalTime){

  dim3 gridSize, blockSize;
  make1dconf(solver->mComp[X]->len, &gridSize, &blockSize); ///@todo cache in heun struct
  
  timer_start("gpuheun_step");{
  
    _gpu_heunstage1<<<gridSize, blockSize>>>(solver->       mComp[X]->list,  solver->      mComp[Y]->list,  solver->      mComp[Z]->list,
                                             solver->       hComp[X]->list,  solver->      hComp[Y]->list,  solver->      hComp[Z]->list,
                                             solver-> torque0Comp[X]->list,  solver->torque0Comp[Y]->list,  solver->torque0Comp[Z]->list,
                                             solver->      m0Comp[X]->list,  solver->     m0Comp[Y]->list,  solver->     m0Comp[Z]->list,
                                             solver->params->hExt[X],       solver->params->hExt[Y],        solver->params->hExt[Z],
                                             0.5f * solver->params->maxDt, solver->params->alpha);
    gpu_sync();
  
  }timer_stop("gpuheun_step");
}


void gpu_heun_step(gpuheun* solver, tensor* m, tensor* h, double* totalTime){

  solver->m->list = m->list;
  solver->h->list = h->list;
  for(int i=0; i<3; i++){
    solver->mComp[i]->list = &(solver->m->list[i * solver->mComp[X]->len]);
    solver->hComp[i]->list = &(solver->h->list[i * solver->hComp[X]->len]);
  }
  
  
  if(solver->stage == 0){
    tensor_copy_on_gpu(solver->m, solver->m0);
    gpuheun_stage0(solver, m, h, totalTime);
    *totalTime += 0.5 * solver->params->maxDt;
    solver->stage++;
  }
  else{
    gpuheun_stage1(solver, m, h, totalTime);
    *totalTime += 0.5 * solver->params->maxDt;
    solver->stage = 0;
  }
}



gpuheun* new_gpuheun(param* p){
  
  check_param(p);
  gpuheun* heun = (gpuheun*)malloc(sizeof(gpuheun));

  heun->stage = 0;
  heun->params = p;
  
  int* size4D = tensor_size4D(p->size);
  assert(size4D[0] == 3);
  int* kernelSize = p->kernelSize;

  heun->m       = as_tensorN(NULL, 4, size4D);
  heun->m0      = new_gputensor(4, size4D);
  heun->h       = as_tensorN(NULL, 4, size4D);
  heun->torque0 = new_gputensor(4, size4D);

  for(int i=0; i<3; i++){
    heun->      mComp[i] = tensor_component(heun->m,       i);
    heun->     m0Comp[i] = tensor_component(heun->m0,      i);
    heun->      hComp[i] = tensor_component(heun->h,       i);
    heun->torque0Comp[i] = tensor_component(heun->torque0, i);
  }

  return heun;
}


// gpuheun* new_gpuheun(int* size, tensor* kernel, float* hExt){
//   
//   gpuheun* heun = (gpuheun*)malloc(sizeof(gpuheun));
// 
//   heun->params = NULL;
//   
//   int* size4D = tensor_size4D(size);
//   assert(size4D[0] == 3);
//   int* kernelSize = (int*)safe_calloc(3, sizeof(int));
//   kernelSize[X] = kernel->size[2+X];
//   kernelSize[Y] = kernel->size[2+Y];
//   kernelSize[Z] = kernel->size[2+Z];
// 
//   fprintf(stderr, "new_gpuheun([%d x %d x %d],[%d x %d x %d],[%g, %g, %g])\n", size[X], size[Y], size[Z], kernelSize[X], kernelSize[Y], kernelSize[Z], hExt[X], hExt[Y], hExt[Z]);
//   
//   heun->m       = new_gputensor(4, size4D);
//   heun->m0      = new_gputensor(4, size4D);
//   heun->h       = new_gputensor(4, size4D);
//   heun->torque0 = new_gputensor(4, size4D);
//   
//   for(int i=0; i<3; i++){
//     heun->      mComp[i] = tensor_component(heun->m,       i);
//     heun->     m0Comp[i] = tensor_component(heun->m0,      i);
//     heun->      hComp[i] = tensor_component(heun->h,       i);
//     heun->torque0Comp[i] = tensor_component(heun->torque0, i);
//   }
//   
//   heun->convplan = new_gpuconv2(size, kernelSize);
//   gpuconv2_loadkernel5DSymm(heun->convplan, kernel);
// 
//   
//   fprintf(stderr, "new_gpuheun(): OK\n");
//   return heun;
// }

#ifdef __cplusplus
}
#endif