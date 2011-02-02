#include "cpu_kernmul.h"
#include "assert.h"
#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

// |Hx|   |Kxx Kxy Kxz|   |Mx|
// |Hy| = |Kxy Kyy Kyz| * |My|
// |Hz|   |Kxz Kyz Kzz|   |Mz|

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



// |Hx|   |Kxx  0   0 |   |Mx|
// |Hy| = | 0  Kyy Kyz| * |My|
// |Hz|   | 0  Kyz Kzz|   |Mz|

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



// |Hx|   | 0  0   0 |   |Mx|
// |Hy| = | 0 Kyy Kyz| * |My|
// |Hz|   | 0 Kyz Kzz|   |Mz|

typedef struct{
  float *fftMy, *fftMz, *fftKyy, *fftKzz, *fftKyz;
  int nRealNumbers;
}cpu_kernelmul3_arg;

void cpu_kernelmul3_t(int id){

  cpu_kernelmul3_arg *arg = (cpu_kernelmul3_arg *) func_arg;
  
  int start, stop;
  init_start_stop (&start, &stop, id, arg->nRealNumbers/2);

  for(int i=start; i<stop; i++){
    int e = i * 2;

    float reMy = arg->fftMy[e  ];
    float imMy = arg->fftMy[e+1];

    float reMz = arg->fftMz[e  ];
    float imMz = arg->fftMz[e+1];

    float Kyy = arg->fftKyy[i];
    float Kyz = arg->fftKyz[i];
    float Kzz = arg->fftKzz[i];

    arg->fftMy[e  ] = reMy * Kyy + reMz * Kyz;
    arg->fftMy[e+1] = imMy * Kyy + imMz * Kyz;
    arg->fftMz[e  ] = reMy * Kyz + reMz * Kzz;
    arg->fftMz[e+1] = imMy * Kyz + imMz * Kzz;

  }
  
  return;
}

void cpu_kernelmul3(float* fftMy,  float* fftMz,
                    float* fftKyy, float* fftKzz, float* fftKyz,
                    int nRealNumbers){
  
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  cpu_kernelmul3_arg args;
  args.fftMy = fftMy;
  args.fftMz = fftMz;
  args.fftKyy = fftKyy;
  args.fftKzz = fftKzz;
  args.fftKyz = fftKyz;
  args.nRealNumbers = nRealNumbers;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_kernelmul3_t);
  
  return;
}



// |Hx|   | 0   Kz -Ky|   |Jx|
// |Hy| = |-Kz  0   Kx| * |Jy|
// |Hz|   | Ky -Kx  0 |   |Jz|

typedef struct{
  float *fftJx, *fftJy, *fftJz, *fftKx, *fftKy, *fftKz;
  int nRealNumbers;
}cpu_kernelmul_biot_savart3D_arg;

void cpu_kernelmul_biot_savart3D_t(int id){

  cpu_kernelmul_biot_savart3D_arg *arg = (cpu_kernelmul_biot_savart_3D_arg *) func_arg;
  
  int start, stop;
  init_start_stop (&start, &stop, id, arg->nRealNumbers/2);

  for(int i=start; i<stop; i++){
    int e = i * 2;
    float reJx = arg->fftJx[e  ];
    float imJx = arg->fftJx[e+1];

    float reJy = arg->fftJy[e  ];
    float imJy = arg->fftJy[e+1];

    float reJz = arg->fftJz[e  ];
    float imJz = arg->fftJz[e+1];

    float Kx = arg->fftKx[i];
    float Ky = arg->fftKy[i];
    float Kz = arg->fftKz[i];

    arg->fftJx[e  ] =  reJy * Kz - reJz * Ky;
    arg->fftJx[e+1] =  imJy * Kz - imJz * Ky;

    arg->fftJy[e  ] = -reJx * Kz + reJz * Kx;
    arg->fftJy[e+1] = -imJx * Kz + imJz * Kx;

    arg->fftJz[e  ] =  reJx * Ky - reJy * Kx;
    arg->fftJz[e+1] =  imJx * Ky - imJy * Kx;

  }
  
  return;
}

void cpu_kernelmul_biot_savart3D(float* fftJx, float* fftJy, float* fftJz,
                                 float* fftKx, float* fftKy, float* fftKz,
                                 int nRealNumbers){
  
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  cpu_kernelmul_biot_savart3D_arg args;
  args.fftJx = fftJx;
  args.fftJy = fftJy;
  args.fftJz = fftJz;
  args.fftKx = fftKx;
  args.fftKy = fftKy;
  args.fftKz = fftKz;
  args.nRealNumbers = nRealNumbers;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_kernelmul_biot_savart3D_t);
  
  return;
}



// |Hx|   | 0   Kz -Ky|   |Jx|
// |Hy| = |-Kz  0   0 | * |Jy|
// |Hz|   | Ky  0   0 |   |Jz|

typedef struct{
  float *fftJx, *fftJy, *fftJz, *fftKy, *fftKz;
  int nRealNumbers;
}cpu_kernelmul_biot_savart3D_Nx1_arg;

void cpu_kernelmul_biot_savart3D_Nx1_t(int id){

  cpu_kernelmul_biot_savart3D_Nx1_arg *arg = (cpu_kernelmul_biot_savart_3D_Nx1_arg *) func_arg;
  
  int start, stop;
  init_start_stop (&start, &stop, id, arg->nRealNumbers/2);

  for(int i=start; i<stop; i++){
    int e = i * 2;
    float reJx = arg->fftJx[e  ];
    float imJx = arg->fftJx[e+1];

    float reJy = arg->fftJy[e  ];
    float imJy = arg->fftJy[e+1];

    float reJz = arg->fftJz[e  ];
    float imJz = arg->fftJz[e+1];

    float Ky = arg->fftKy[i];
    float Kz = arg->fftKz[i];

    arg->fftJx[e  ] =  reJy * Kz - reJz * Ky;
    arg->fftJx[e+1] =  imJy * Kz - imJz * Ky;

    arg->fftJy[e  ] = -reJx * Kz;
    arg->fftJy[e+1] = -imJx * Kz;

    arg->fftJz[e  ] =  reJx * Ky;
    arg->fftJz[e+1] =  imJx * Ky;

  }
  
  return;
}

void cpu_kernelmul_biot_savart3D_Nx1(float* fftJx, float* fftJy, float* fftJz,
                                     float* fftKy, float* fftKz,
                                     int nRealNumbers){
  
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  cpu_kernelmul_biot_savart3D_arg args;
  args.fftJx = fftJx;
  args.fftJy = fftJy;
  args.fftJz = fftJz;
  args.fftKy = fftKy;
  args.fftKz = fftKz;
  args.nRealNumbers = nRealNumbers;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_kernelmul_biot_savart3D_Nx1_t);
  
  return;
}



// |Hx|   | 0   0  0|   |Jx|
// |Hy| = |-Kz  0  0| * | 0|
// |Hz|   | Ky  0  0|   | 0|

typedef struct{
  float *fftJx, *fftJy, *fftJz, *fftKy, *fftKz;
  int nRealNumbers;
}cpu_kernelmul_biot_savart2D_arg;

void cpu_kernelmul_biot_savart2D_t(int id){

  cpu_kernelmul_biot_savart2D_arg *arg = (cpu_kernelmul_biot_savart_2D_arg *) func_arg;
  
  int start, stop;
  init_start_stop (&start, &stop, id, arg->nRealNumbers/2);

  for(int i=start; i<stop; i++){
    int e = i * 2;

    float reJx = arg->fftJx[e  ];
    float imJx = arg->fftJx[e+1];

    float Ky = arg->fftKy[i];
    float Kz = arg->fftKz[i];

    arg->fftJy[e  ] = -reJx * Kz;
    arg->fftJy[e+1] = -imJx * Kz;

    arg->fftJz[e  ] =  reJx * Ky;
    arg->fftJz[e+1] =  imJx * Ky;

  }
  
  return;
}

void cpu_kernelmul_biot_savart2D(float* fftJx, float* fftJy, float* fftJz,
                                 float* fftKy, float* fftKz,
                                 int nRealNumbers){
  
  assert(nRealNumbers > 0);
  assert(nRealNumbers % 2 == 0);

  cpu_kernelmul_biot_savart2D_arg args;
  args.fftJx = fftJx;
  args.fftJy = fftJy;
  args.fftJz = fftJz;
  args.fftKy = fftKy;
  args.fftKz = fftKz;
  args.nRealNumbers = nRealNumbers;

  func_arg = (void *) (&args);

  thread_Wrapper(cpu_kernelmul_biot_savart2D_t);
  
  return;
}


#ifdef __cplusplus
}
#endif
