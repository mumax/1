#include "cpu_kernmul.h"
#include "assert.h"
#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
  float *fftMx, *fftMy, *fftMz, *fftKxx, *fftKyy, *fftKzz, *fftKyz, *fftKxz, *fftKxy;
  int nRealNumbers;
}cpu_kernelmul6_arg;


void cpu_kernelmul6_t(int id){

  cpu_kernelmul6_arg *arg = (cpu_kernelmul6_arg *) func_arg;
  
  int start, stop;
  init_start_stop (&start, &stop, id, arg->nRealNumbers/2);

  for(int i=start; i<stop; i++){
    int e = i * 2;
    float reMx = arg->fftMx[e  ];
    float imMx = arg->fftMx[e+1];

    float reMy = arg->fftMy[e  ];
    float imMy = arg->fftMy[e+1];

    float reMz = arg->fftMz[e  ];
    float imMz = arg->fftMz[e+1];

    float Kxx = arg->fftKxx[i];
    float Kyy = arg->fftKyy[i];
    float Kzz = arg->fftKzz[i];

    float Kyz = arg->fftKyz[i];
    float Kxz = arg->fftKxz[i];
    float Kxy = arg->fftKxy[i];

    arg->fftMx[e  ] = reMx * Kxx + reMy * Kxy + reMz * Kxz;
    arg->fftMx[e+1] = imMx * Kxx + imMy * Kxy + imMz * Kxz;

    arg->fftMy[e  ] = reMx * Kxy + reMy * Kyy + reMz * Kyz;
    arg->fftMy[e+1] = imMx * Kxy + imMy * Kyy + imMz * Kyz;

    arg->fftMz[e  ] = reMx * Kxz + reMy * Kyz + reMz * Kzz;
    arg->fftMz[e+1] = imMx * Kxz + imMy * Kyz + imMz * Kzz;

  }
  
  return;
}

void cpu_kernelmul6(float* fftMx,  float* fftMy,  float* fftMz,
                    float* fftKxx, float* fftKyy, float* fftKzz,
                    float* fftKyz, float* fftKxz, float* fftKxy,
                    int nRealNumbers){
  
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  cpu_kernelmul6_arg args;
  args.fftMx = fftMx;
  args.fftMy = fftMy;
  args.fftMz = fftMz;
  args.fftKxx = fftKxx;
  args.fftKyy = fftKyy;
  args.fftKzz = fftKzz;
  args.fftKyz = fftKyz;
  args.fftKxz = fftKxz;
  args.fftKxy = fftKxy;
  args.nRealNumbers = nRealNumbers;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_kernelmul6_t);
  
  return;
}




typedef struct{
  float *fftMx, *fftMy, *fftMz, *fftKxx, *fftKyy, *fftKzz, *fftKyz;
  int nRealNumbers;
}cpu_kernelmul4_arg;


void cpu_kernelmul4_t(int id){

  cpu_kernelmul4_arg *arg = (cpu_kernelmul4_arg *) func_arg;
  
  int start, stop;
  init_start_stop (&start, &stop, id, arg->nRealNumbers/2);

  for(int i=start; i<stop; i++){
    int e = i * 2;
    float reMx = arg->fftMx[e  ];
    float imMx = arg->fftMx[e+1];

    float reMy = arg->fftMy[e  ];
    float imMy = arg->fftMy[e+1];

    float reMz = arg->fftMz[e  ];
    float imMz = arg->fftMz[e+1];

    float Kxx = arg->fftKxx[i];
    float Kyy = arg->fftKyy[i];
    float Kzz = arg->fftKzz[i];

    float Kyz = arg->fftKyz[i];

    arg->fftMx[e  ] = reMx * Kxx;
    arg->fftMx[e+1] = imMx * Kxx;
    arg->fftMy[e  ] = reMy * Kyy + reMz * Kyz;
    arg->fftMy[e+1] = imMy * Kyy + imMz * Kyz;
    arg->fftMz[e  ] = reMy * Kyz + reMz * Kzz;
    arg->fftMz[e+1] = imMy * Kyz + imMz * Kzz;
  }
}


void cpu_kernelmul4(float* fftMx,  float* fftMy,  float* fftMz,
                    float* fftKxx, float* fftKyy, float* fftKzz,
                    float* fftKyz,
                    int nRealNumbers){
  
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  cpu_kernelmul4_arg args;
  args.fftMx = fftMx;
  args.fftMy = fftMy;
  args.fftMz = fftMz;
  args.fftKxx = fftKxx;
  args.fftKyy = fftKyy;
  args.fftKzz = fftKzz;
  args.fftKyz = fftKyz;
  args.nRealNumbers = nRealNumbers;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_kernelmul4_t);

  return;
}

#ifdef __cplusplus
}
#endif
