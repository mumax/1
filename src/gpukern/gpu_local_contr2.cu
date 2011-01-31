#include "gpu_local_contr2.h"
#include "gpu_mem.h"
#include "gpu_conf.h"
#include "gpu_safe.h"
#include "../macros.h"

#ifdef __cplusplus
extern "C" {
#endif

__global__ void _gpu_add_local_fields_uniaxial(float* mx, float* my, float* mz,
                                              float* hx, float* hy, float* hz,
                                              float hext_x, float hext_y, float hext_z,
                                              float k_x2, float U0, float U1, float U2,
                                              int N){
  int i = threadindex;

  if(i<N){
    float mu = mx[i] * U0 + my[i] * U1 + mz[i] * U2;

    hx[i] += hext_x + k_x2 * mu * U0;
    hy[i] += hext_y + k_x2 * mu * U1;
    hz[i] += hext_z + k_x2 * mu * U2;
    
  }
}


__global__ void _gpu_add_external_field(float* hx, float* hy, float* hz,
                                        float hext_x, float hext_y, float hext_z,
                                        int N){
  int i = threadindex;

  if(i<N){
    hx [i] += hext_x;
    hy [i] += hext_y;
    hz [i] += hext_z;
  }
}


void gpu_add_local_fields (float* m, float* h, int N, float* Hext, int anisType, float* anisK, float* anisAxes){


  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  float* hx = &(h[0*N]);
  float* hy = &(h[1*N]);
  float* hz = &(h[2*N]);

  /*
    Uniaxial anisotropy:
    H_anis = ( K_1 / (mu0 Ms) )  ( m . u ) u
	u = axis, normalized
  */
  
  dim3 gridsize, blocksize;
  make1dconf(N, &gridsize, &blocksize);

  switch (anisType){
    default: abort();
    case ANIS_NONE:
       _gpu_add_external_field<<<gridsize, blocksize>>>(hx, hy, hz,  Hext[X], Hext[Y], Hext[Z],  N);
       break;
    case ANIS_UNIAXIAL:
	  //printf("anis: K, u,: %f  %f,%f,%f  \n", anisK[0],anisAxes[0],anisAxes[1],anisAxes[2]);
      _gpu_add_local_fields_uniaxial<<<gridsize, blocksize>>>(mx, my, mz,
                                                             hx, hy, hz,
                                                             Hext[X], Hext[Y], Hext[Z],
                                                             anisK[0],  anisAxes[0], anisAxes[1], anisAxes[2], N);
      break;
  }
  gpu_sync();
}

#ifdef __cplusplus
}
#endif
