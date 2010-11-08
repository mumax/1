#include "gpu_anis.h"
#include "gpu_conf.h"

#ifdef __cplusplus
extern "C" {
#endif


__global__ void _gpu_add_lin_anisotropy(float* hx, float* hy, float* hz,
                                        float* mx, float* my, float* mz,
                                        float* kxx, float* kyy, float* kzz,
                                        float* kyz, float* kxz, float* kxy,
                                        int N){
  int i = threadindex;

  if(i < N){
    float Mx = mx[i];
    float My = my[i];
    float Mz = mz[i];

    float Kxx = kxx[i];
    float Kyy = kyy[i];
    float Kzz = kzz[i];
    float Kyz = kyz[i];
    float Kxz = kxz[i];
    float Kxy = kxy[i];

    hx[i] += Mx * Kxx + My * Kxy + Mz * Kxz;
    hy[i] += Mx * Kxy + My * Kyy + Mz * Kyz;
    hz[i] += Mx * Kxz + My * Kyz + Mz * Kzz;
  }
}


void gpu_add_lin_anisotropy(float* hx, float* hy, float* hz,
                            float* mx, float* my, float* mz,
                            float* kxx, float* kyy, float* kzz,
                            float* kyz, float* kxz, float* kxy,
                            int N){
  
}
                            

#ifdef __cplusplus
}
#endif
