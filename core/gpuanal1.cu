#include "gpuanal1.h"
#include "timer.h"
#include <stdio.h>
#include <assert.h>

#ifdef __cplusplus
extern "C" {
#endif

#define alpha 1.0f


__global__ void _gpu_anal1step(float* mx, float* my, float* mz, float* hx, float* hy, float* hz, float dt){

	int i = ((blockIdx.x * blockDim.x) + threadIdx.x);
	float rot[9], hxy_r, hxyz_r;

// 	if (mx[i]==0.0f && my[i]==0.0f && *mz[i]==0.0f)
// 		continue;

	if (hx[i]==0.0f && hy[i] ==0.0f){
		rot[0] = 0.0f;
		rot[1] = 0.0f;
		rot[2] = -1.0f;
		rot[3] = 0.0f;
		rot[4] = 1.0f;
		rot[5] = 0.0f;
		rot[6] = 1.0f;
//			rot[7] = 0.0f;
		rot[8] = 0.0f;

		hxyz_r = 1.0f/hz[i];
	}
	else{
		float temp = hx[i]*hx[i] + hy[i]*hy[i];
		hxy_r = rsqrtf(temp);
		hxyz_r = rsqrtf(temp + hz[i]*hz[i]);

		rot[0] = hx[i]*hxyz_r;
		rot[1] = - hy[i]*hxy_r;
		rot[2] = - rot[0]*hz[i]*hxy_r;
		rot[3] = hy[i]*hxyz_r;
		rot[4] = hx[i]*hxy_r;
		rot[5] = rot[1]*hz[i]*hxyz_r;
		rot[6] = hz[i]*hxyz_r;
//			rot[7] = 0.0f;
		rot[8] = hxyz_r/hxy_r;
	}

	float mx_rot = mx[i]*rot[0] + my[i]*rot[3] + mz[i]*rot[6];
	float my_rot = mx[i]*rot[1] + my[i]*rot[4];
	float mz_rot = mx[i]*rot[2] + my[i]*rot[5] + mz[i]*rot[8];

// te checken wegens normering constanten!!
	float at = dt / (1+alpha*alpha);
	float act = alpha*at;
// ----------------------------------------

	float ex, sn, cs, denom;
	ex = exp(act/hxyz_r);
	__sincosf(at/hxyz_r, &sn, &cs);
	denom = ex*(1.0f+mx_rot) + (1.0f-mx_rot)/ex;

	float mx_rotnw = (ex*(1.0f+mx_rot) - (1.0f-mx_rot)/ex)/denom;
	float my_rotnw = 2.0f*(my_rot*cs - mz_rot*sn)/denom;
	float mz_rotnw = 2.0f*(my_rot*sn + mz_rot*cs)/denom;

// in deze lijnen komt fout tot uiting
/*	mx[i] = mx_rotnw*rot[0] + my_rotnw*rot[1] + mz_rotnw*rot[2];
	my[i] = mx_rotnw*rot[3] + my_rotnw*rot[4] + mz_rotnw*rot[5];
	mz[i] = mx_rotnw*rot[6] + mz_rotnw*rot[8];*/
// -----------------------------------


// wanneer volgende ipv vorige lijnen uitgevoerd worden is alles ok (met fout resultaat natuurlijk)
  mx[i] = mx_rotnw;
  my[i] = my_rotnw;
  mz[i] = mz_rotnw;
// -----------------------------------------------------------------------------------------------

	//correction on possible accumulating errors on amplitude M, should not be done frequently
	float norm = rsqrtf(mx[i]*mx[i] + my[i]*my[i] + mz[i]*mz[i]);
  mx[i] *= norm;
  my[i] *= norm;
  mz[i] *= norm;
}



void gpuanal1_step(gpuanal1* solver, float dt){
  int threadsPerBlock = 512;
  gpuconv1_exec(solver->convplan, solver->m, solver->h);
  int blocks = (solver->convplan->len_m_comp) / threadsPerBlock;
  gpu_checkconf_int(blocks, threadsPerBlock);
  timer_start("gpuanal1_step");
  _gpu_anal1step<<<blocks, threadsPerBlock>>>(solver->convplan->m_comp[0], solver->convplan->m_comp[1], solver->convplan->m_comp[2], solver->convplan->h_comp[0], solver->convplan->h_comp[1], solver->convplan->h_comp[2], dt);
  cudaThreadSynchronize();
  timer_stop("gpuanal1_step");
}

void gpuanal1_checksize_m(gpuanal1* sim, tensor* m){
   // m should be a rank 4 tensor with size 3 x N0 x N1 x N2
  assert(m->rank == 4);
  assert(m->size[0] == 3); 
  for(int i=0; i<3; i++){ assert(m->size[i+1] == sim->size[i]); }
}

void gpuanal1_loadm(gpuanal1* anal1, tensor* m){
  gpuanal1_checksize_m(anal1, m); 
  memcpy_to_gpu(m->list, anal1->m, anal1->len_m);
}

void gpuanal1_storem(gpuanal1* anal1, tensor* m){
  gpuanal1_checksize_m(anal1, m); 
  memcpy_from_gpu(anal1->m, m->list, anal1->len_m);
}

void gpuanal1_init_sizes(gpuanal1* anal1, int N0, int N1, int N2){
  anal1->size = (int*)calloc(3, sizeof(int));
  anal1->size[0] = N0; anal1->size[1] = N1; anal1->size[2] = N2;
  anal1->N = N0 * N1 * N2;
}

void gpuanal1_init_m(gpuanal1* anal1){
  anal1->len_m = 3 * anal1->N;
  anal1->m = new_gpu_array(anal1->len_m);
}

void gpuanal1_init_h(gpuanal1* anal1){
  anal1->len_h = 3 * anal1->N;
  anal1->h = new_gpu_array(anal1->len_h);
}

gpuanal1* new_gpuanal1(int N0, int N1, int N2, tensor* kernel){
  gpuanal1* anal1 = (gpuanal1*)malloc(sizeof(gpuanal1));
  gpuanal1_init_sizes(anal1, N0, N1, N2);
  gpuanal1_init_m(anal1);
  gpuanal1_init_h(anal1);
  anal1->convplan = new_gpuconv1(N0, N1, N2, kernel);
  return anal1;
}

#ifdef __cplusplus
}
#endif