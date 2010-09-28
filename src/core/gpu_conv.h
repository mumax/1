/**
 * @file
 * This file contains all functions related with the initialization and execution of the convolutions.
 * 
 * @author Ben Van de Wiele, Arne Vansteenkiste
 */
#ifndef GPU_CONV_H
#define GPU_CONV_H

#include "gpukern.h"
#include "tensor.h"
#include "param.h"
#include "kernel.h"
//#include "gpufft2.h"
#include "gpu_fft.h"
#include "assert.h"
#include "timer.h"
#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
  
  gpuFFT3dPlan *fftplan;   ///< FFT plan adapted to the size of the simulation.
  tensor *fft1;            ///< buffer to store and transform the zero-padded magnetization and field.
  tensor *fft2;            ///< second fft buffer. By default, this one points to fft1, so everything is in-place.
  tensor *kernel;          ///< kernel used in the convolution.
  
}conv_data;


/**
 * Evaluates a convolution.  The input is the magnetization tensor m, the output is the magnetic field tensor h. 
 * All required work space, the kernel and the corresponding FFT plan are referred ate in the convolution data 'conv'. 
 * Before using this function, the convolution data should be initialized using the function: 
 * new_conv_data(param *p, tensor *kernel).
 */
void evaluate_convolution(tensor *m,         ///< input: rank 4 tensor containing the magnetization data.
                          tensor *h,         ///< output: rank 4 tensor containing the field data.
                          conv_data *conv,   ///< convolution data: work space, kernel.
                          param *p           ///< parameter list
                          );



// evaluation of the micromag3d convolution ***********************************************************
/**
 * Evaluates a convolution of type MICROMAG3D with a sample thickness of more then one (possibly coarse level)
 * FD cell.  
 */
void evaluate_micromag3d_conv(tensor *m,         ///< input: rank 4 tensor containing the magnetization data.
                              tensor *h,         ///< output: rank 4 tensor containing the field data.
                              conv_data *conv    ///< convolution data: work space, kernel, FFT plan.
                              );

/**
 * Evaluates the kernel multiplication for the MICROMAG3D kernel with a sample thickness of more then one 
 * (possibly coarse level) FD cell.  
 **/
void gpu_kernel_mul_micromag3d(tensor *fft1,     ///< Fourier transformed data (as input: m-data, when exiting: h-data)
                               tensor *kernel    ///< MICROMAG3d kernel
                               );

/**
 * Actual multiplication on gpu of the MICROMAG3D kernel with a sample thickness of more then one 
 * (possibly coarse level) FD cell.
 **/
__global__ void _gpu_kernel_mul_micromag3d(float *fftMx,    ///< Fourier transformed Mx data
                                           float *fftMy,    ///< Fourier transformed My data  
                                           float *fftMz,    ///< Fourier transformed Mz data
                                           float *fftKxx,   ///< xx kernel component
                                           float *fftKxy,   ///< xy kernel component
                                           float *fftKxz,   ///< xz kernel component
                                           float *fftKyy,   ///< yy kernel component
                                           float *fftKyz,   ///< yz kernel component
                                           float *fftKzz    ///< zz kernel component
                                           );
// ****************************************************************************************************



// evaluation of the micromag3d convolution with thickness 1 FD cell **********************************
/**
 * Evaluates a convolution of type MICROMAG3D with a sample thickness of one (possibly coarse level) FD cell.  
 */
void evaluate_micromag3d_conv_Xthickness_1(tensor *m,         ///< input: rank 4 tensor containing the magnetization data.
                                           tensor *h,         ///< output: rank 4 tensor containing the field data.
                                           conv_data *conv    ///< convolution data: work space, kernel, FFT plan.
                                           );

/**
 * Evaluates the kernel multiplication for the MICROMAG3D kernel with a sample thickness of one (possibly 
 * coarse level) FD cell.  
 **/
void gpu_kernel_mul_micromag3d_Xthickness_1(tensor *fft1,     ///< Fourier transformed data (as input: m-data, when exiting: h-data)
                                            tensor *kernel    ///< MICROMAG3d kernel
                                            );

/**
 * Actual multiplication on gpu of the MICROMAG3D kernel with a sample thickness of one (possibly coarse level) 
 * FD cell.
 **/
__global__ void _gpu_kernel_mul_micromag3d_Xthickness_1(float *fftMx,    ///< Fourier transformed Mx data
                                                        float *fftMy,    ///< Fourier transformed My data
                                                        float *fftMz,    ///< Fourier transformed Mz data
                                                        float *fftKxx,   ///< xx kernel component
                                                        float *fftKyy,   ///< yy kernel component
                                                        float *fftKyz,   ///< yz kernel component
                                                        float *fftKzz    ///< zz kernel component
                                                        );
// ****************************************************************************************************



// evaluation of the micromag2d convolution ***********************************************************
/**
 * Evaluates a convolution of type MICROMAG2D.  
 */
void evaluate_micromag2d_conv(tensor *m,         ///< input: rank 4 tensor containing the magnetization data.
                              tensor *h,         ///< output: rank 4 tensor containing the field data.
                              conv_data *conv    ///< convolution data: work space, kernel, FFT plan.
                              );

void gpu_kernel_mul_micromag2d(tensor *fft1,     ///< Fourier transformed data (as input: m-data, when exiting: h-data)
                               tensor *kernel    ///< MICROMAG3d kernel
                               );

__global__ void _gpu_kernel_mul_micromag2d(float *fftMy,    ///< Fourier transformed My data
                                           float *fftMz,    ///< Fourier transformed Mz data
                                           float *fftKyy,   ///< yy kernel component
                                           float *fftKyz,   ///< yz kernel component
                                           float *fftKzz    ///< zz kernel component
                                           );
// ****************************************************************************************************



// to initialize the convolution **********************************************************************
/**
 * Initializes the convolution data based on the parameter list p and links to the initialized kernel.
 * Before using this function, the kernel data should be initialized using the function: new_kernel(param *p).
 **/
conv_data *new_conv_data(param *p,            ///< parameter list
                         tensor *kernel       ///< tensor containing the kernel data
                         );
// ****************************************************************************************************




// // functions for copying to and from padded matrix ****************************************************
// /**
//  * @internal Does padding and unpadding, not necessarily by a factor 2
//  **/
// __global__ void _gpu_copy_pad(float* source,        ///< source data
//                               float* dest,          ///< destination data
//                               int S1,               ///< source size Y
//                               int S2,               ///< source size Z
//                               int D1,               ///< destination size Y
//                               int D2                ///< destination size Z
//                               );
//                               
// /**
//  * @internal
//  * pads the input tensor 'source' and saves it in tensor 'dest', 2d and 3d applicable.
//  */
// void gpu_copy_to_pad(float* source,         ///< input: unpadded source as contiguous float array
//                      float* dest,           ///< output: padded destination as contiguous float array
//                      int *unpad_size4d,     ///< size of the corresponding unpadded tensor 
//                      int *pad_size4d        ///< size of the corresponding padded tensor
//                      );
// 
// /**
//  * @internal
//  * copies the non-zero elements from the padded the input tensor 'source' towards tensor 'dest', 2d and 3d applicable.
//  */
// void gpu_copy_to_unpad(float* source,        ///< input: padded source as contiguous float array
//                        float* dest,          ///< output: unpadded destination as contiguous float array
//                        int *pad_size4d,      ///< size of the corresponding padded tensor
//                        int *unpad_size4d     ///< size of the corresponding unpadded tensor 
//                        ); 
// 



#ifdef __cplusplus
}
#endif
#endif