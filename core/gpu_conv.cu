#include "gpu_conv.h"

#ifdef __cplusplus
extern "C" {
#endif

void evaluate_convolution(tensor *m, tensor *h, conv_data *conv, param *p){

  switch (p->kernelType){
    case KERNEL_MICROMAG3D:
      if (p->size[X]>1)  
        evaluate_micromag3d_conv(m, h, conv);
      if (p->size[X]==1)
        evaluate_micromag3d_conv_Xthickness_1(m, h, conv);
      break;
    case KERNEL_MICROMAG2D:
      evaluate_micromag3d_conv(m, h, conv);
      break;
    default:
      fprintf(stderr, "abort: no valid kernelType\n");
      abort;
  }

  return;
}



// evaluation of the micromag3d convolution ***********************************************************
void evaluate_micromag3d_conv(tensor *m, tensor *h, conv_data *conv){

  int m_length = m->len/3;
  int fft_length = conv->fft1->len/3;

  float m_comp[3], h_comp[3], fft1_comp[3];
  for (int i=0; i<3; i++){
    fft1_comp[i] = &conv->fft1->list[i*fft_length];
    m_comp[i]    = &m->list[i*m_length];
    h_comp[i]    = &h->list[i*m_length];
  }

    // zero out fft1
  gpu_zero_tensor(fft1);
  
  for(int i=0; i<3; i++){
      //padding of m_i
    gpu_copy_to_pad(m_comp[i], fft1_comp[i]);
      //Fourier transforming of fft_mi
    gpuFFT3dPlan_forward(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
  }

    // kernel multiplication
  gpu_kernel_mul_micromag3d(conv->fft1, kernel);

  for(int i=0; i<3; i++){
      //inverse Fourier transforming fft_hi
    gpuFFT3dPlan_inverse(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
      //unpadding of fft_hi
    gpu_copy_to_unpad(fft1_comp[i], h_comp[i]);
  }

  return;
}

void gpu_kernel_mul_micromag3d(tensor *fft1, tensor *kernel){
  
  int fft_length = conv->fft1->len/3;
  int gridSize = -1;
  int blockSize = -1;
  make1dconf(fft_length/2, &gridSize, &blockSize);

  float *fftMx = &fft1->list[0*fft_length];
  float *fftMy = &fft1->list[1*fft_length];
  float *fftMz = &fft1->list[2*fft_length];
  float *fftKxx = &kernel->list[0*fft_length/2];
  float *fftKxy = &kernel->list[1*fft_length/2];
  float *fftKxz = &kernel->list[2*fft_length/2];
  float *fftKyy = &kernel->list[3*fft_length/2];
  float *fftKyz = &kernel->list[4*fft_length/2];
  float *fftKzz = &kernel->list[5*fft_length/2];

  _gpu_kernel_mul_micromag3d <<<gridSize, blockSize>>> (fftMx, fftMy, fftMz, fftKxx, fftKxy, fftKxz, fftKyy, fftKyz, fftKzz);
  cudaThreadSynchronize();
  
  return;
}

__global__ void _gpu_kernel_mul_micromag3d(float* fftMx,  float* fftMy,  float* fftMz, 
                                           float* fftKxx, float* fftKxy, float* fftKxz,
                                           float* fftKyy, float* fftKyz, float* fftKzz){
  
  int e = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
  
  // we use shared memory here, which saves an "8N" buffer in the global memory
  ///@todo coalescale read/writes
  float reMx = fftMx[e  ];
  float imMx = fftMx[e+1];

  float reMy = fftMy[e  ];
  float imMy = fftMy[e+1];

  float reMz = fftMz[e  ];
  float imMz = fftMz[e+1];

  float Kxx = fftKxx[e];
  float Kxy = fftKxy[e];
  float Kxz = fftKxz[e];
  float Kyy = fftKyy[e];
  float Kyz = fftKyz[e];
  float Kzz = fftKzz[e];
  
  fftMx[e  ] = Kxx*reMx + Kxy*reMy + Kxz*reMz;
  fftMx[e+1] = Kxx*imMx + Kxy*imMy + Kxz*imMz;
  fftMy[e  ] = Kxy*reMx + Kyy*reMy + Kyz*reMz;
  fftMy[e+1] = Kxy*imMx + Kyy*imMy + Kyz*imMz;
  fftMz[e  ] = Kxz*reMx + Kyz*reMy + Kzz*reMz;
  fftMz[e+1] = Kxz*imMx + Kyz*imMy + Kzz*imMz;

  return;
}
// ****************************************************************************************************




// evaluation of the micromag3d convolution with thickness 1 FD cell **********************************
void evaluate_micromag3d_conv_Xthickness_1(tensor *m, tensor *h, conv_data *conv){

  int m_length = m->len/3;
  int fft_length = conv->fft1->len/3;

  float m_comp[3], h_comp[3], fft1_comp[3];
  for (int i=0; i<3; i++){
    fft1_comp[i] = &conv->fft1->list[i*fft_length];
    m_comp[i]    = &m->list[i*m_length];
    h_comp[i]    = &h->list[i*m_length];
  }

    // zero out fft1
  gpu_zero_tensor(fft1);
  
  for(int i=0; i<3; i++){
      //padding of m_i
    gpu_copy_to_pad(m_comp[i], fft1_comp[i]);
      //Fourier transforming of fft_mi
    gpuFFT3dPlan_forward(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
  }

    // kernel multiplication
  gpu_kernel_mul_micromag3d_Xthickness_1(conv->fft1, kernel);

  for(int i=0; i<3; i++){
      //inverse Fourier transforming fft_hi
    gpuFFT3dPlan_inverse(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
      //unpadding of fft_hi
    gpu_copy_to_unpad(fft1_comp[i], h_comp[i]);
  }

  return;
}

void gpu_kernel_mul_micromag3d_Xthickness_1(tensor *fft1, tensor *kernel){
  
  int fft_length = conv->fft1->len/3;
  int gridSize = -1;
  int blockSize = -1;
  make1dconf(fft_length/2, &gridSize, &blockSize);

  float *fftMx = &fft1->list[0*fft_length];
  float *fftMy = &fft1->list[1*fft_length];
  float *fftMz = &fft1->list[2*fft_length];
  float *fftKxx = &kernel->list[0*fft_length/2];
  float *fftKyy = &kernel->list[1*fft_length/2];
  float *fftKyz = &kernel->list[2*fft_length/2];
  float *fftKzz = &kernel->list[3*fft_length/2];

  _gpu_kernel_mul_micromag3d <<<gridSize, blockSize>>> (fftMx, fftMy, fftMz, fftKxx, fftKyy, fftKyz, fftKzz);
  cudaThreadSynchronize();
  
  return;
}

__global__ void _gpu_kernel_mul_micromag3d_Xthickness_1(float* fftMx,  float* fftMy,  float* fftMz, 
                                                        float* fftKxx, float* fftKyy, float* fftKyz, float* fftKzz){
  
  int e = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
  
  // we use shared memory here, which saves an "8N" buffer in the global memory
  ///@todo coalescale read/writes
  float reMx = fftMx[e  ];
  float imMx = fftMx[e+1];

  float reMy = fftMy[e  ];
  float imMy = fftMy[e+1];

  float reMz = fftMz[e  ];
  float imMz = fftMz[e+1];

  float Kxx = fftKxx[e];
  float Kyy = fftKyy[e];
  float Kyz = fftKyz[e];
  float Kzz = fftKzz[e];
  
  fftMx[e  ] = Kxx*reMx;
  fftMx[e+1] = Kxx*imMx;
  fftMy[e  ] = Kyy*reMy + Kyz*reMz;
  fftMy[e+1] = Kyy*imMy + Kyz*imMz;
  fftMz[e  ] = Kyz*reMy + Kzz*reMz;
  fftMz[e+1] = Kyz*imMy + Kzz*imMz;

  return;
}
// ****************************************************************************************************



// evaluation of the micromag3d convolution ***********************************************************
void evaluate_micromag2d_conv(tensor *m, tensor *h, conv_data *conv){

  int m_length = m->len/3;
  int fft_length = conv->fft1->len/2;    // only 2 components need to be convolved!

  float m_comp[3], h_comp[3], fft1_comp[3];
  for (int i=1; i<3; i++){
    fft1_comp[i-1] = &conv->fft1->list[(i-1)*fft_length];
    m_comp[i-1]    = &m->list[i*m_length];
    h_comp[i-1]    = &h->list[i*m_length];
  }

    // zero out fft1
  gpu_zero_tensor(fft1);
  
  for(int i=0; i<2; i++){
      //padding of m_i
    gpu_copy_to_pad(m_comp[i], fft1_comp[i]);
      //Fourier transforming of fft_mi
    gpuFFT3dPlan_forward(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
  }

    // kernel multiplication
  gpu_kernel_mul_micromag2d(conv->fft1, kernel);

  for(int i=0; i<3; i++){
      //inverse Fourier transforming fft_hi
    gpuFFT3dPlan_inverse(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
      //unpadding of fft_hi
    gpu_copy_to_unpad(fft1_comp[i], h_comp[i]);
  }

  return;
}

void gpu_kernel_mul_micromag2d(tensor *fft1, tensor *kernel){
  
  int fft_length = conv->fft1->len/3;
  int gridSize = -1;
  int blockSize = -1;
  make1dconf(fft_length/2, &gridSize, &blockSize);

  float *fftMy = &fft1->list[0*fft_length];
  float *fftMz = &fft1->list[1*fft_length];
  float *fftKyy = &kernel->list[0*fft_length/2];
  float *fftKyz = &kernel->list[1*fft_length/2];
  float *fftKzz = &kernel->list[2*fft_length/2];

  _gpu_kernel_mul_micromag3d <<<gridSize, blockSize>>> (fftMy, fftMz, fftKyy, fftKyz, fftKzz);
  cudaThreadSynchronize();
  
  return;
}

__global__ void _gpu_kernel_mul_micromag3d(float* fftMy, float* fftMz, float* fftKyy, float* fftKyz, float* fftKzz){
  
  int e = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
  
  // we use shared memory here, which saves an "8N" buffer in the global memory
  ///@todo coalescale read/writes
  float reMy = fftMy[e  ];
  float imMy = fftMy[e+1];

  float reMz = fftMz[e  ];
  float imMz = fftMz[e+1];

  float Kyy = fftKyy[e];
  float Kyz = fftKyz[e];
  float Kzz = fftKzz[e];
  
  fftMy[e  ] = Kyy*reMy + Kyz*reMz;
  fftMy[e+1] = Kyy*imMy + Kyz*imMz;
  fftMz[e  ] = Kyz*reMy + Kzz*reMz;
  fftMz[e+1] = Kyz*imMy + Kzz*imMz;

  return;
}
// ****************************************************************************************************



// functions for copying to and from padded matrix ****************************************************
conv_data *new_conv_data(param *p, tensor *kernel){

  conv_data *conv = (conv_data *) calloc(1, sizeof(conv));

  switch (p->kernelType){
    case KERNEL_MICROMAG3D:
      int size4d[4] = {3, p->kernelSize[X], p->kernelSize[Y], gpu_pad_to_stride(p->kernelSize[Z]+2};
      break;
    case KERNEL_MICROMAG2D:
      int size4d[4] = {2, p->kernelSize[X], p->kernelSize[Y], gpu_pad_to_stride(p->kernelSize[Z]+2};
      break;
    default:
      fprintf(stderr, "abort: no valid kernelType\n");
      abort;
  }

  conv->fft1 = new_gputensor(4, size4d);
  conv->fft2 = conv_fft1;
  conv->plan = new_gpuFFT3dPlan_padded(p->size, p->kernelSize);
  conv->kernel = kernel;

  return (conv);
}
// ****************************************************************************************************




// functions for copying to and from padded matrix ****************************************************
/// @internal Does padding and unpadding, not necessarily by a factor 2
__global__ void _gpu_copy_pad(float* source, float* dest, 
                                   int S1, int S2,                  ///< source sizes Y and Z
                                   int D1, int D2                   ///< destination size Y and Z
                                   ){
  int i = blockIdx.x;
  int j = blockIdx.y;
  int k = threadIdx.x;

  dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
  
  return;
}


void gpu_copy_to_pad(tensor* source, tensor* dest){        //for padding of the tensor, 2d and 3d applicable
  
  assert(source->rank == 3);
  assert(  dest->rank == 3);
  
  // source must not be larger than dest
  for(int i=0; i<3; i++){
    assert(source->size[i] <= dest->size[i]);
  }
  
  int S0 = source->size[X];
  int S1 = source->size[Y];
  int S2 = source->size[Z];

  dim3 gridSize(S0, S1, 1); ///@todo generalize!
  dim3 blockSize(S2, 1, 1);
  gpu_checkconf(gridSize, blockSize);

  _gpu_copy_pad<<<gridSize, blockSize>>>(source->list, dest->list, S1, S2, dest->size[Y], dest->size[Z]);
  cudaThreadSynchronize();
  
  return;
}

void gpu_copy_to_unpad(tensor* source, tensor* dest){        //for unpadding of the tensor, 2d and 3d applicable
  
  assert(source->rank == 3);
  assert(  dest->rank == 3);
  
  // dest must not be larger than source
  for(int i=0; i<3; i++){
    assert(source->size[i] >= dest->size[i]);
  }
  
  int D0 = dest->size[X];
  int D1 = dest->size[Y];
  int D2 = dest->size[Z];

  dim3 gridSize(D0, D1, 1); ///@todo generalize!
  dim3 blockSize(D2, 1, 1);
  gpu_checkconf(gridSize, blockSize);

  _gpu_copy_pad<<<gridSize, blockSize>>>(source->list, dest->list, source->size[1], source->size[2], D1, D2);
  cudaThreadSynchronize();
  
  return;
}
// ****************************************************************************************************


#ifdef __cplusplus
}
#endif