/**
 * @file
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef gpu_fft3_h
#define gpu_fft3_h

#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif


/**
 * A real-to-complex FFT plan on the GPU.
 */
typedef struct{
  int* size;               ///< logical size of the (real, unpadded) input data
  int  dataN;              ///< total number of floats in dataSize

  int* paddedSize;         ///< size after zero-padding. @note zero-padding is conditional and not necessarily performed in each direction. Therefore, "paddedSize" is not necessarily twice "dataSize".
  int  paddedN;            ///< total number of floats in paddedSize

//   int* paddedComplexSize;  ///< physical size of the (complex, padded) output data in half-complex format
//   int  paddedComplexN;
  
  cufftHandle fwPlanZ;     ///< 1D real-to-complex plan for Z-direction
  cufftHandle invPlanZ;    ///< 1D complex-to-real plan for Z-direction
  cufftHandle planY;       ///< 1D complex-to-complex plan for Y-direction, forward or inverse
  cufftHandle planX;       ///< 1D complex-to-complex plan for X-direction, forward or inverse

  float* buffer1;          ///< Buffer for zero-padding in Z
  float* buffer2;          ///< Buffer for result of out-of-place FFT_z
  float* buffer2t;         ///< buffer2 transposed
  float* buffer3;          ///< Buffer for zero-padding in Y and in-place transform
  float* buffer3t;         ///< buffer3 transposed

}gpuFFT3dPlan;



/**
 * Creates a new real-to-complex 3D FFT plan with efficient handling of padding zeros.
 * If paddedsize is larger than size, then the additional space is filled with zeros,
 * but they are efficiently handled during the transform.
 */
gpuFFT3dPlan* new_gpuFFT3dPlan_padded(int* size,       ///< size of real input data (3D)
                                      int* paddedsize  ///< size of the padded data (3D). Should be at least the size of the input data. If the kernel is larger, the input data is assumed to be padded with zero's which are efficiently handled by the FFT
                                      );

/**
 * @internal
 * Forward (real-to-complex) transform.
 * Sizes are not checked.
 * @see gpuFFT3dPlan_inverse()
 */
void gpuFFT3dPlan_forward(gpuFFT3dPlan* plan,      ///< the plan to be executed
                          float* input,            ///< input data. Size = dataSize. Real
                          float* output            ///< output data. Size = paddedComplexSize. Half-complex format
                          );

/**
 * @internal
 * Backward (complex-to-real) transform.
 * Sizes are not checked.
 * @see gpuFFT3dPlan_forward()
 */
void gpuFFT3dPlan_inverse(gpuFFT3dPlan* plan,       ///< the plan to be executed
                                 float* input,      ///< input data, Size = paddedComplexSize. Half-complex format
                                 float* output      ///< output data, Size = dataSize. Real
                                 );
                                 
///@todo access to X,Y,Z transforms for multi GPU / MPI implementation

#ifdef __cplusplus
}
#endif
#endif
