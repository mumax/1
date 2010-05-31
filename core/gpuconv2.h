/**
 * @file
 * A smarter vector convolution plan on the GPU:
 * real-to-complex FFT's.
 * The redundant zero's in the padded magnetization buffers are ignored.
 * The zero's in the micromagnetic kernel are ignored.
 * Care is taken to align CUDA memory access.
 *
 * The interface is flexible: gpuconv2_exec(m, h) can be called on any 
 * magnetization and field array that match the size of the plan. 
 * m and h are thus not stored in the plan itself. 
 * This is handy for higher order solvers that keep multiple versions of m and h.
 *
 * @todo Should we use tensors everywhere ?
 *
 * @see gpuconv1, new_gpuconv2, gpuconv2_exec
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef GPUCONV2_H
#define GPUCONV2_H

#include "tensor.h"
#include "gputil.h"
#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif

//_________________________________________________________________________________________ FFT

/**
 * A real-to-complex FFT plan on the GPU.
 */
typedef struct{  
  int* size;		///< logical size of the (real) input data
  int N;		///< total number of floats in size
  
  int* paddedSize;	///< size after zero-padding. @note zero-padding is conditional and not necessarily performed in each direction. Therefore, "paddedSize" is not necessarily twice "size".
  int paddedN;		///< total number of floats in paddedSize
  
  int* paddedStorageSize;	///< A real-to-complex FFT requires padding with one complex number in the last dimension. However, is this would result in misalgned memory, we pad with (typically) 64 floats
  int paddedStorageN;		///< total number of floats in paddedStorageSize
  
  cufftHandle fwPlanZ;	///< 1D real-to-complex plan for Z-direction
  cufftHandle invPlanZ;	///< 1D complex-to-real plan for Z-direction
  cufftHandle planY;	///< 1D complex-to-complex plan for Y-direction, forward or inverse
  cufftHandle planX;	///< 1D complex-to-complex plan for X-direction, forward or inverse
  
  float* transp;	///< buffer for out-of-place transposing
  
}gpu_plan3d_real_input;


/**
 * Creates a new FFT plan for transforming the magnetization. 
 * Zero-padding in each dimension is optional, and rows with
 * only zero's are not transformed.
 */
gpu_plan3d_real_input* new_gpu_plan3d_real_input(int N0,	///< size of real input data in x-direction
						int N1,		///< size of real input data  in y-direction
						int N2,		///< size of real input data  in z-direction
						int* zero_pad	///< 3 ints, should be 1 or 0, meaning zero-padding or no zero-padding in X,Y,Z respectively
						);

/**
 * Executes in-place.
 */
void gpu_plan3d_real_input_forward(gpu_plan3d_real_input* plan,	///< the plan to be executed
		        float* data	///< data to be transformed in-place, it size should be plan->paddedStorageSize
			);

/**
 * Executes in-place.
 */
void gpu_plan3d_real_input_inverse(gpu_plan3d_real_input* plan,	///< the plan to be executed
		        float* data	///< data to be transformed in-place, it size should be plan->paddedStorageSize
			);

/**
 * Swaps the Y and Z components of a 3D array of complex numbers.
 * N0 x N1 x N2/2 complex numbers are stored as N0 x N1 x N2 interleaved real numbers.
 */
void gpu_transposeYZ_complex(float* source, 
			     float* dest,
			     int N0,
			     int N1, 
			     int N2 ///< number of floats (!) in the Z-direction, thus 2x the number of complex numbers in Z.
			     );
/**
 * @see gpu_transposeYZ_complex()
 */
void gpu_transposeXZ_complex(float* source, 
			     float* dest, 
			     int N0, 
			     int N1, 
			     int N2
			     );

/**
 * Frees the FFT plan
 */
void delete_gpu_plan3d_real_input(gpu_plan3d_real_input* plan	///< the plan to be deleted
		      );

		      
//_________________________________________________________________________________________ convolution

/**
 * 
 */
typedef struct{
  
   /*int* size;							///< 3D size of the magnetization field
   int N;									///< total number of magnetization vectors for linear access
   
   int* paddedSize;		///< 3D size of the zero-padded magnetization buffer
   int paddedN;			///< total number of magnetization vectors in the padded magnetization buffer, for linear access
   
   
   int* paddedStorageSize;	///< 3D size of the zero-padded magnetization buffer, in complex-number format
  int paddedComplexN;		///< total number of magnetization vectors in the padded magnetization buffer in complex-number format, for linear access*/

   int len_m;					///< total number of floats in the magnetization array
   int len_m_comp;				///< total number of floats in each of the m_comp array (1/3 of len_m)
   float* ft_m_i;				///< buffer for one componet of m, zero-padded and in complex-format 
   int len_ft_m_i;				///< total number of floats in ft_m_i

   float*** ft_kernel;		///< ft_kernel[s][d] gives the d-component of the field of a a unit vector along the s direction (in Fourier space). These components are themselves 3D fields of size paddedComplexSize. 
   int len_ft_kernel;
   int len_ft_kernel_ij;
   int len_kernel_ij;
   
   int len_h;
   int len_h_comp;
   float* ft_h;			///< buffer for the FFT'ed magnetic field
   int len_ft_h;
   float** ft_h_comp;		///< points to X, Y and Z components of ft_h
   int len_ft_h_comp;
   
   gpu_plan3d_real_input* fftplan;
  
}gpuconv2;



/**
 * New convolution plan.
 * 
 */
gpuconv2* new_gpuconv2(int N0,		///< X size of the magnetization vector field
		       int N1,		///< Y size of the magnetization vector field
		       int N2,  	///< Z size of the magnetization vector field
		       tensor* kernel,	///< convolution kernel of size 3 x 3 x 2*N0 x 2*N1 x 2*N2
		       int* zero_pad
		       );

/**
 * Executes the convolution plan: convolves the source data with the stored kernel and stores the result in the destination pointer.
 */
void gpuconv2_exec(gpuconv2* plan,	///< the plan to execute 
		   float* source, 	///< the input vector field (magnetization)
		   float* dest	///< the destination vector field (magnetic field) to store the result in
		   );

/**
 * Loads a kernel. Automatically called during new_gpuconv2(), but could be used to change the kernel afterwards.
 * @see new_gpuconv2
 */
void gpuconv2_loadkernel(gpuconv2* plan,	///< plan to load the kernel into
			 tensor* kernel		///< kernel to load (should match the plan size)
			 );

/**
 * Pointwise multiplication of arrays of complex numbers. ft_h_comp_j += ft_m_i * ft_kernel_comp_ij. Runs on the GPU.
 * Makes use of kernel symmetry
 * @todo store in texture memory (writing is slow but only done once, reading is very fast)
 */
void gpu_kernel_mul2(float* ft_m_i,		///< multiplication input 1
		     float* ft_kernel_comp_ij, 	///< multiplication input 2
		     float* ft_h_comp_j, 	///< multiplication result gets added to this array
		     int nRealNumbers		///< the number of floats(!) in each of the arrays, thus twice the number of complex's in them.
		     );


#ifdef __cplusplus
}
#endif
#endif