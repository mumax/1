#include "gpu_conv.h"
#include "gputil.h"

#ifdef __cplusplus
extern "C" {
#endif

void evaluate_convolution(tensor *m, tensor *h, conv_data *conv, param *p){

  for (int i=0; i<3; i++)
    if (p->demagCoarse[i]>1){
      fprintf(stderr, "abort: convolution on a coarse grid not yet implemented.\n");
      abort();
    }

  switch (p->kernelType){
    case KERNEL_MICROMAG3D:
      if (p->size[X]/p->demagCoarse[X] > 1)
        evaluate_micromag3d_conv(m, h, conv);
      if (p->size[X]/p->demagCoarse[X] == 1)
        evaluate_micromag3d_conv_Xthickness_1(m, h, conv);
      break;
    case KERNEL_MICROMAG2D:
      evaluate_micromag2d_conv(m, h, conv);
      break;
    default:
      fprintf(stderr, "abort: no valid kernelType\n");
      abort();
  }

  return;
}



// evaluation of the micromag3d convolution ***********************************************************
void evaluate_micromag3d_conv(tensor *m, tensor *h, conv_data *conv){

  int m_length = m->len/3;
  int fft_length = conv->fft1->len/3;

  float *m_comp[3], *h_comp[3], *fft1_comp[3];
  for (int i=0; i<3; i++){
    fft1_comp[i] = &conv->fft1->list[i*fft_length];
    m_comp[i]    = &m->list[i*m_length];
    h_comp[i]    = &h->list[i*m_length];
  }

     // zero out fft1
  gpu_zero_tensor(conv->fft1);
  
//   for(int i=0; i<3; i++){
//       //padding of m_i
//     gpu_copy_to_pad(m_comp[i], fft1_comp[i], m->size, conv->fft1->size);
//       //Fourier transforming of fft_mi
//     gpuFFT3dPlan_forward_unsafe(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
//   }

  for(int i=0; i<3; i++){
/*      //padding of m_i
    gpu_copy_to_pad(m_comp[i], fft1_comp[i], m->size, conv->fft1->size);*/
      //Fourier transforming of fft_mi
    gpuFFT3dPlan_forward_unsafe(conv->fftplan, m_comp[i], fft1_comp[i]);    //padding within routine
  }

    // kernel multiplication
  gpu_kernel_mul_micromag3d(conv->fft1, conv->kernel);

  for(int i=0; i<3; i++){
      //inverse Fourier transforming fft_hi
    gpuFFT3dPlan_inverse_unsafe(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
      //unpadding of fft_hi
    gpu_copy_to_unpad(fft1_comp[i], h_comp[i], conv->fft1->size, m->size);
  }

  return;
}

void gpu_kernel_mul_micromag3d(tensor *fft1, tensor *kernel){
  
  int fft_length = fft1->len/3;
  dim3 gridSize, blockSize;
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
  gpu_sync();
  
  return;
}

__global__ void _gpu_kernel_mul_micromag3d(float* fftMx,  float* fftMy,  float* fftMz, 
                                           float* fftKxx, float* fftKxy, float* fftKxz,
                                           float* fftKyy, float* fftKyz, float* fftKzz){
  
  int e = 2 * (blockIdx.x * blockDim.x + threadIdx.x);
  
  // we use shared memory here, which saves an "8N" buffer in the global memory
  ///@todo coalescale read/writes
  float reMx = fftMx[e  ];
  float imMx = fftMx[e+1];

  float reMy = fftMy[e  ];
  float imMy = fftMy[e+1];

  float reMz = fftMz[e  ];
  float imMz = fftMz[e+1];

  float Kxx = fftKxx[e/2];
  float Kxy = fftKxy[e/2];
  float Kxz = fftKxz[e/2];
  float Kyy = fftKyy[e/2];
  float Kyz = fftKyz[e/2];
  float Kzz = fftKzz[e/2];
  
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

  float *m_comp[3], *h_comp[3], *fft1_comp[3];
  for (int i=0; i<3; i++){
    fft1_comp[i] = &conv->fft1->list[i*fft_length];
    m_comp[i]    = &m->list[i*m_length];
    h_comp[i]    = &h->list[i*m_length];
  }

    // zero out fft1
  gpu_zero_tensor(conv->fft1);
  
  for(int i=0; i<3; i++){
/*      //padding of m_i
    gpu_copy_to_pad(m_comp[i], fft1_comp[i], m->size, conv->fft1->size);*/
      //Fourier transforming of fft_mi
    gpuFFT3dPlan_forward_unsafe(conv->fftplan, m_comp[i], fft1_comp[i]);    //padding within routine
  }
//   for(int i=0; i<3; i++){
//       //padding of m_i
//     gpu_copy_to_pad(m_comp[i], fft1_comp[i], m->size, conv->fft1->size);
//       //Fourier transforming of fft_mi
//     gpuFFT3dPlan_forward_unsafe(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
//   }

    // kernel multiplication
  gpu_kernel_mul_micromag3d_Xthickness_1(conv->fft1, conv->kernel);

  for(int i=0; i<3; i++){
      //inverse Fourier transforming fft_hi
    gpuFFT3dPlan_inverse_unsafe(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
      //unpadding of fft_hi
    gpu_copy_to_unpad(fft1_comp[i], h_comp[i], conv->fft1->size, m->size);
  }

  return;
}

void gpu_kernel_mul_micromag3d_Xthickness_1(tensor *fft1, tensor *kernel){
  
  int fft_length = fft1->len/3;

  dim3 gridSize, blockSize;
  make1dconf(fft_length/2, &gridSize, &blockSize);
  
  float *fftMx = &fft1->list[0*fft_length];
  float *fftMy = &fft1->list[1*fft_length];
  float *fftMz = &fft1->list[2*fft_length];
  float *fftKxx = &kernel->list[0*fft_length/2];
  float *fftKyy = &kernel->list[1*fft_length/2];
  float *fftKyz = &kernel->list[2*fft_length/2];
  float *fftKzz = &kernel->list[3*fft_length/2];
  
  _gpu_kernel_mul_micromag3d_Xthickness_1 <<<gridSize, blockSize>>> (fftMx, fftMy, fftMz, fftKxx, fftKyy, fftKyz, fftKzz);
  gpu_sync();

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

  float Kxx = fftKxx[e/2];
  float Kyy = fftKyy[e/2];
  float Kyz = fftKyz[e/2];
  float Kzz = fftKzz[e/2];
  
  fftMx[e  ] = Kxx*reMx;
  fftMx[e+1] = Kxx*imMx;
  fftMy[e  ] = Kyy*reMy + Kyz*reMz;
  fftMy[e+1] = Kyy*imMy + Kyz*imMz;
  fftMz[e  ] = Kyz*reMy + Kzz*reMz;
  fftMz[e+1] = Kyz*imMy + Kzz*imMz;

  return;
}

// ****************************************************************************************************



// evaluation of the micromag2d convolution ***********************************************************
void evaluate_micromag2d_conv(tensor *m, tensor *h, conv_data *conv){

  int m_length = m->len/3;
  int fft_length = conv->fft1->len/2;    // only 2 components need to be convolved!

  float *m_comp[2], *h_comp[2], *fft1_comp[2];
  for (int i=1; i<3; i++){
    fft1_comp[i-1] = &conv->fft1->list[(i-1)*fft_length];
    m_comp[i-1]    = &m->list[i*m_length];
    h_comp[i-1]    = &h->list[i*m_length];
  }

    // zero out fft1
  gpu_zero_tensor(conv->fft1);
  
/*  for(int i=0; i<2; i++){
      //padding of m_i
    gpu_copy_to_pad(m_comp[i], fft1_comp[i], m->size, conv->fft1->size);
      //Fourier transforming of fft_mi
    gpuFFT3dPlan_forward_unsafe(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
  }*/
  for(int i=0; i<2; i++){
/*      //padding of m_i
    gpu_copy_to_pad(m_comp[i], fft1_comp[i], m->size, conv->fft1->size);*/
      //Fourier transforming of fft_mi
    gpuFFT3dPlan_forward_unsafe(conv->fftplan, m_comp[i], fft1_comp[i]);    //padding within routine
  }

    // kernel multiplication
  gpu_kernel_mul_micromag2d(conv->fft1, conv->kernel);

  for(int i=0; i<2; i++){
      //inverse Fourier transforming fft_hi
    gpuFFT3dPlan_inverse_unsafe(conv->fftplan, fft1_comp[i], fft1_comp[i]);  ///@todo out-of-place
      //unpadding of fft_hi
    gpu_copy_to_unpad(fft1_comp[i], h_comp[i], conv->fft1->size, m->size);
  }

  return;
}

void gpu_kernel_mul_micromag2d(tensor *fft1, tensor *kernel){
  
  int fft_length = fft1->len/3;
  dim3 gridSize, blockSize;
  make1dconf(fft_length/2, &gridSize, &blockSize);

  float *fftMy = &fft1->list[0*fft_length];
  float *fftMz = &fft1->list[1*fft_length];
  float *fftKyy = &kernel->list[0*fft_length/2];
  float *fftKyz = &kernel->list[1*fft_length/2];
  float *fftKzz = &kernel->list[2*fft_length/2];

  _gpu_kernel_mul_micromag2d <<<gridSize, blockSize>>> (fftMy, fftMz, fftKyy, fftKyz, fftKzz);
  gpu_sync();
  
  return;
}

__global__ void _gpu_kernel_mul_micromag2d(float* fftMy, float* fftMz, float* fftKyy, float* fftKyz, float* fftKzz){
  
  int e = 2 * ((blockIdx.x * blockDim.x) + threadIdx.x);
  
  // we use shared memory here, which saves an "8N" buffer in the global memory
  ///@todo coalescale read/writes
  float reMy = fftMy[e  ];
  float imMy = fftMy[e+1];

  float reMz = fftMz[e  ];
  float imMz = fftMz[e+1];

  float Kyy = fftKyy[e/2];
  float Kyz = fftKyz[e/2];
  float Kzz = fftKzz[e/2];
  
  fftMy[e  ] = Kyy*reMy + Kyz*reMz;
  fftMy[e+1] = Kyy*imMy + Kyz*imMz;
  fftMz[e  ] = Kyz*reMy + Kzz*reMz;
  fftMz[e+1] = Kyz*imMy + Kzz*imMz;

  return;
}
// ****************************************************************************************************



// functions for copying to and from padded matrix ****************************************************
conv_data *new_conv_data(param *p, tensor *kernel){

  ///@todo add a test that checks if the kernel has been initialized.   
  conv_data *conv = (conv_data *) calloc(1, sizeof(conv));
  int size4d[4] = {0, p->kernelSize[X], p->kernelSize[Y], gpu_pad_to_stride(p->kernelSize[Z]+2)};
  
  switch (p->kernelType){
    case KERNEL_MICROMAG3D:
      size4d[0] = 3;
      break;
    case KERNEL_MICROMAG2D:
      size4d[0] = 2;
      break;
    default:
      fprintf(stderr, "abort: no valid kernelType\n");
      abort();
  }

  conv->fft1 = new_gputensor(4, size4d);
  conv->fft2 = conv->fft1;
  conv->fftplan = new_gpuFFT3dPlan_padded(p->size, p->kernelSize);
  conv->kernel = kernel;

  return (conv);
}
// ****************************************************************************************************




// // functions for copying to and from padded matrix ****************************************************
// /// @internal Does padding and unpadding, not necessarily by a factor 2
// __global__ void _gpu_copy_pad(float* source, float* dest, 
//                                    int S1, int S2,                  ///< source sizes Y and Z
//                                    int D1, int D2                   ///< destination size Y and Z
//                                    ){
//   int i = blockIdx.x;
//   int j = blockIdx.y;
//   int k = threadIdx.x;
// 
//   dest[(i*D1 + j)*D2 + k] = source[(i*S1 + j)*S2 + k];
//   
//   return;
// }
// 
// 
// 
// 
// void gpu_copy_to_pad(float* source, float* dest, int *unpad_size4d, int *pad_size4d){          //for padding of the tensor, 2d and 3d applicable
//   
//   int S0 = unpad_size4d[1];
//   int S1 = unpad_size4d[2];
//   int S2 = unpad_size4d[3];
// 
//   dim3 gridSize(S0, S1, 1); ///@todo generalize!
//   dim3 blockSize(S2, 1, 1);
//   gpu_checkconf(gridSize, blockSize);
//   
//   if ( pad_size4d[1]!=unpad_size4d[1] || pad_size4d[2]!=unpad_size4d[2])
//     _gpu_copy_pad<<<gridSize, blockSize>>>(source, dest, S1, S2, S1, pad_size4d[3]-2);      // for out of place forward FFTs in z-direction, contiguous data arrays
//   else
//     _gpu_copy_pad<<<gridSize, blockSize>>>(source, dest, S1, S2, S1, pad_size4d[3]);        // for in place forward FFTs in z-direction, contiguous data arrays
// 
//   gpu_sync();
//   
//   return;
// }
// 
// void gpu_copy_to_unpad(float* source, float* dest, int *pad_size4d, int *unpad_size4d){        //for unpadding of the tensor, 2d and 3d applicable
//   
//   int D0 = unpad_size4d[1];
//   int D1 = unpad_size4d[2];
//   int D2 = unpad_size4d[3];
// 
//   dim3 gridSize(D0, D1, 1); ///@todo generalize!
//   dim3 blockSize(D2, 1, 1);
//   gpu_checkconf(gridSize, blockSize);
// 
//   if ( pad_size4d[1]!=unpad_size4d[1] || pad_size4d[2]!=unpad_size4d[2])
//     _gpu_copy_pad<<<gridSize, blockSize>>>(source, dest, D1,  pad_size4d[3]-2, D1, D2);       // for out of place inverse FFTs in z-direction, contiguous data arrays
//   else
//     _gpu_copy_pad<<<gridSize, blockSize>>>(source, dest, D1,  pad_size4d[3], D1, D2);         // for in place inverse FFTs in z-direction, contiguous data arrays
// 
//     gpu_sync();
//   
//   return;
// }
// // ****************************************************************************************************


#ifdef __cplusplus
}
#endif