#include "cpu_local_contr.h"
#include "../macros.h"
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void cpu_add_local_fields_uniaxial(float* mx, float* my, float* mz,
                                              float* hx, float* hy, float* hz,
                                              float hext_x, float hext_y, float hext_z,
                                              float U0, float U1, float U2,
                                              int N){
  #pragma omp parallel for
  for(int i = 0; i < N; i++){
    float mu = mx[i] * U0 + my[i] * U1 + mz[i] * U2;

    hx[i] += hext_x + mu * U0;
    hy[i] += hext_y + mu * U1;
    hz[i] += hext_z + mu * U2;

  }
}


void cpu_add_external_field(float* hx, float* hy, float* hz,
                                        float hext_x, float hext_y, float hext_z,
                                        int N){
  #pragma omp parallel for
  for(int i = 0; i < N; i++){
    hx [i] += hext_x;
    hy [i] += hext_y;
    hz [i] += hext_z;
  }
}


void cpu_add_local_fields (float* m, float* h, int N, float* Hext, int anisType, float* anisK, float* anisAxes){


  float* mx = &(m[0*N]);
  float* my = &(m[1*N]);
  float* mz = &(m[2*N]);

  float* hx = &(h[0*N]);
  float* hy = &(h[1*N]);
  float* hz = &(h[2*N]);

  /*
    Uniaxial anisotropy:
    H_anis = ( 2K_1 / (mu0 Ms) )  ( m . u ) u
    U := sqrt( 2K_1 / (mu0 Ms) )
    H_anis = (m . U) U
  */
  float U0, U1, U2;

  switch (anisType){
    default: abort();
    case ANIS_NONE:
       cpu_add_external_field(hx, hy, hz,  Hext[X], Hext[Y], Hext[Z],  N);
       break;
    case ANIS_UNIAXIAL:
      U0 = sqrt(2.0 * anisK[0]) * anisAxes[0];
      U1 = sqrt(2.0 * anisK[0]) * anisAxes[1];
      U2 = sqrt(2.0 * anisK[0]) * anisAxes[2];
      cpu_add_local_fields_uniaxial(mx, my, mz,
                                    hx, hy, hz,
                                    Hext[X], Hext[Y], Hext[Z],
                                    U0, U1, U2, N);
      break;
  }
}


#ifdef __cplusplus
}
#endif
