#include "gpu_anal.h"
#include "gpukern.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

void gpu_anal_fw_step(param *p, tensor *m_in, tensor *m_out, tensor *h){

  int length = m_in->len/3;
  
  int gridSize = -1, blockSize = -1;
  make1dconf(length, &gridSize, &blockSize); ///@todo cache in gpu_anal struct
  
  timer_start("gpu_anal_fw_step");
  _gpu_anal_fw_step <<<gridSize, blockSize>>> (&m_in->list[X*length], &m_in->list[Y*length], &m_in->list[Z*length], &m_out->list[X*length], &m_out->list[Y*length], &m_out->list[Z*length], &h->list[X*length], &h->list[Y*length], &h->list[Z*length], p->maxDt, p->alpha);
  gpu_sync();
  timer_stop("gpu_anal_fw_step");
  

  return;
}

__global__ void _gpu_anal_fw_step (float *minx, float *miny, float *minz, float *moutx, float *mouty, float *moutz, float *hx, float *hy, float *hz, float dt, float alpha){
	
	int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
	float hxy_r, hxyz_r;
	float rot0, rot1, rot2, rot3, rot4, rot5, rot6, rot8;

// 	if (mx[i]==0.0f && my[i]==0.0f && *mz[i]==0.0f)
// 		continue;

	if (hx[i]==0.0f && hy[i] ==0.0f){
		rot0 = 0.0f;
		rot1 = 0.0f;
		rot2 = -1.0f;
		rot3 = 0.0f;
		rot4 = 1.0f;
		rot5 = 0.0f;
		rot6 = 1.0f;
//			rot[7] = 0.0f;
		rot8 = 0.0f;

		hxyz_r = 1.0f/hz[i];
	}
	else{
		float temp = hx[i]*hx[i] + hy[i]*hy[i];
		hxy_r = rsqrtf(temp);
		hxyz_r = rsqrtf(temp + hz[i]*hz[i]);

		rot0 = hx[i]*hxyz_r;
		rot1 = - hy[i]*hxy_r;
//    rot1 = - hy[i]*hxyz_r;
		rot2 = - rot0*hz[i]*hxy_r;
//    rot2 = - hx[i]*hz[i]*hxy_r*hxyz_r;
		rot3 = hy[i]*hxyz_r;
		rot4 = hx[i]*hxy_r;
		rot5 = rot1*hz[i]*hxyz_r;
//    rot5 = rot1*hz[i]*hxy_r;
//    rot5 = -hx[i]*hz[i]*hxy_r*hxyz_r;
    rot6 = hz[i]*hxyz_r;
//			rot[7] = 0.0f;
    rot8 = hxyz_r/hxy_r;
 	}

	float mx_rot = minx[i]*rot0 + miny[i]*rot3 + minz[i]*rot6;
	float my_rot = minx[i]*rot1 + miny[i]*rot4;
	float mz_rot = minx[i]*rot2 + miny[i]*rot5 + minz[i]*rot8;

/// @todo check used parameters due to normalization of constants!!
	float qt = dt / (1+alpha*alpha);
  float aqt = alpha*qt;
// ----------------------------------------

	float ex, sn, cs, denom;
	ex = exp(aqt/hxyz_r);
	__sincosf(qt/hxyz_r, &sn, &cs);
	denom = ex*(1.0f+mx_rot) + (1.0f-mx_rot)/ex;

	float mx_rotnw = (ex*(1.0f+mx_rot) - (1.0f-mx_rot)/ex)/denom;
	float my_rotnw = 2.0f*(my_rot*cs - mz_rot*sn)/denom;
	float mz_rotnw = 2.0f*(my_rot*sn + mz_rot*cs)/denom;

	moutx[i] = mx_rotnw*rot0 + my_rotnw*rot1 + mz_rotnw*rot2;
	mouty[i] = mx_rotnw*rot3 + my_rotnw*rot4 + mz_rotnw*rot5;
	moutz[i] = mx_rotnw*rot6 + mz_rotnw*rot8;

//   float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);     // inverse square root
//   mx[i] *= norm;
//   my[i] *= norm;
//   mz[i] *= norm;

	return;
}


void gpu_anal_pc_mean_h(tensor *h1, tensor *h2){

  int gridSize = -1, blockSize = -1;
  make1dconf(h1->len, &gridSize, &blockSize); ///@todo cache in gpu_anal struct

  timer_start("gpu_anal_pc_mean_h");
  _gpu_anal_pc_meah_h <<<gridSize, blockSize>>> (h1->list, h2->list);
  gpu_sync();
  timer_stop("gpu_anal_pc_mean_h");

  return;
}

__global__ void _gpu_anal_pc_meah_h (float *h1, float *h2){
  
  int i = ((blockIdx.x * blockDim.x) + threadIdx.x);

  h1[i] = 0.5f*(h1[i] + h2[i]);
  
  return;
}


gpuanalfw* new_gpuanalfw(param* p){
  
  check_param(p);
  gpuanalfw* anal_fw = (gpuanalfw*) malloc(sizeof(gpuanalfw));
  anal_fw->params = p;
  
  return anal_fw;
}

gpuanalpc* new_gpuanalpc(param* p){
  
  check_param(p);
  int* size4D = tensor_size4D(p->size);
  
  gpuanalpc* anal_pc = (gpuanalpc*) malloc(sizeof(gpuanalpc));
  anal_pc->params = p;
  anal_pc->m2 = new_gputensor(4, size4D);
  anal_pc->h2 = new_gputensor(4, size4D);
  return anal_pc;
}


#ifdef __cplusplus
}
#endif