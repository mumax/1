/**
 * @file
 * This file contains all functions related with the initialization and execution of the convolutions.
 * 
 * @author Ben Van de Wiele, Arne Vansteenkiste
 */
#ifndef GPU_CONV_H
#define GPU_CONV_H

#include "gpu_kernmul.h"
#include "gputil.h"
#include "gpu_conf.h"
#include "tensor.h"
#include "param.h"
#include "kernel.h"
//#include "gpufft2.h"
#include "gpu_fft4.h"
#include "gpu_fftbig.h"
#include "assert.h"
#include "timer.h"
#include <stdio.h>


#ifdef __cplusplus
extern "C" {
#endif

typedef struct{
  
  gpuFFT3dPlan_big *fftplan_big;   ///< FFT plan adapted to the size of the simulation.
  gpuFFT3dPlan *fftplan;   ///< FFT plan adapted to the size of the simulation.
  tensor *fft1;                ///< buffer to store and transform the zero-padded magnetization and field.
  tensor *fft2;                ///< second fft buffer. By default, this one points to fft1, so everything is in-place.
  tensor *kernel;              ///< kernel used in the convolution.
  
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

/**
 * Evaluates a convolution of type MICROMAG3D with a sample thickness of more then one (possibly coarse level)
 * FD cell.  
 */
void evaluate_micromag3d_conv(tensor *m,         ///< input: rank 4 tensor containing the magnetization data.
                              tensor *h,         ///< output: rank 4 tensor containing the field data.
                              conv_data *conv    ///< convolution data: work space, kernel, FFT plan.
                              );
/**
 * Evaluates a convolution of type MICROMAG3D with a sample thickness of one (possibly coarse level) FD cell.  
 */
void evaluate_micromag3d_conv_Xthickness_1(tensor *m,         ///< input: rank 4 tensor containing the magnetization data.
                                           tensor *h,         ///< output: rank 4 tensor containing the field data.
                                           conv_data *conv    ///< convolution data: work space, kernel, FFT plan.
                                           );

/**
 * Evaluates a convolution of type MICROMAG2D.  
 */
void evaluate_micromag2d_conv(tensor *m,         ///< input: rank 4 tensor containing the magnetization data.
                              tensor *h,         ///< output: rank 4 tensor containing the field data.
                              conv_data *conv    ///< convolution data: work space, kernel, FFT plan.
                              );

                              
// to initialize the convolution **********************************************************************
/**
 * Initializes the convolution data based on the parameter list p and links to the initialized kernel.
 * Before using this function, the kernel data should be initialized using the function: new_kernel(param *p).
 **/
conv_data *new_conv_data(param *p,            ///< parameter list
                         tensor *kernel       ///< tensor containing the kernel data
                         );
// ****************************************************************************************************



// to be placed in gpu_kernmul.h
/*void gpu_kernelmul4(float *fftMx, 
                    float *fftMy, 
                    float *fftMz, 
                    float *fftKxx, 
                    float *fftKyy, 
                    float *fftKyz, 
                    float *fftKzz, 
                    int nRealNumbers
                    );

void gpu_kernelmul3(float *fftMy, 
                    float *fftMz, 
                    float *fftKyy, 
                    float *fftKyz, 
                    float *fftKzz, 
                    int nRealNumbers
                    );*/
                    
                    
                    


#ifdef __cplusplus
}
#endif
#endif