/**
 * @file
 * Initialization of the micromagnetic kernel for 2D simulations with invariance considered in 
 * the x-direction of all microstructural properties and all fields.  In the y- and z-direction, 
 * the dimensions should be larger then one (possibly coarse level) FD cell. 
 * This kernel contains the demag and exchange (if wanted) contribution and is adapted to the FFT dimensions 
 * of gpufft.cu. The Kernel components are Fourier transformed and only the real parts are stored 
 * on the device. The correction factor 1/(dimensions FFTs) is included.
 * Parameter demagCoarse controls if the kernel is defined on a coarser discretization level.
 * Parameter exchInConv controls if exchange is included in the kernel
 * 
 * @author Ben Van de Wiele
 */
#ifndef GPU_CONV_H
#define GPU_CONV_H

#include "gputil.h"
#include "tensor.h"
#include "param.h"
#include "kernel.h"
#include "gpufft2.h"
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
 * @internal
 * pads the input tensor 'source' and saves it in tensor 'dest', 2d and 3d applicable.
 */
void gpu_copy_to_pad(tensor* source, tensor* dest);

/**
 * @internal
 * copies the non-zero elements from the padded the input tensor 'source' towards tensor 'dest', 2d and 3d applicable.
 */
void gpu_copy_to_unpad(tensor* source, tensor* dest); 




#ifdef __cplusplus
}
#endif
#endif