/**
 * @file
 * This file implements the simplest possible vector convolution plan on the GPU:
 * All FFT's are complex-to-complex, no advantage is taken from the real input data.
 * The zero's in the padded magnetization buffers are not ignored.
 * The zero's in the micromagnetic kernel are not ignored.
 * No care is taken to align CUDA memory access.
 *
 * The interface is flexible: gpuconv1_exec(m, h) can be called on any magnetization and field array that match the size of the plan. m and h are thus not stored in the plan itself. This is handy for higher order solvers that keep multiple versions of m and h.
 *
 * When more intelligent implementations are made, this one can serve as a comparison
 * for correctness and performance.
 *
 * @see new_gpuconv1, gpuconv1_exec
 *
 * @author Arne Vansteenkiste
 */
#ifndef GPUCONV1_H
#define GPUCONV1_H

#include "tensor.h"
#include "gputil.h"
#include <cufft.h>

#ifdef __cplusplus
extern "C" {
#endif

//_________________________________________________________________________________________ FFT

/**
 * A complex-to-complex FFT plan on the GPU.
 * @see new_gpuc2cplan(), delete_gpuc2cplan(), gpuc2cplan_exec().
 */
typedef struct{  
  cufftHandle handle;
}gpuc2cplan;

/**
 * Creates a new 3D complex-to-complex FFT plan for the GPU.
 * @todo There is a difficulty with real-to-complex FFT's:
 * the last dimension must be made 2 complex numbers larger,
 * but then it does not fit the stride anymore.
 * Extra padding? Out-of-place transform?
 */
gpuc2cplan* new_gpuc2cplan(int N0,	///< size in x-direction
			   int N1,	///< size in y-direction
			   int N2	///< size in z-direction
			   );

/**
 * Forward FFT direction.
 * @see gpuc2cplan_exec()
 */
#define FORWARD	CUFFT_FORWARD


/**
 * Backward FFT direction.
 * @see gpuc2cplan_exec()
 */
#define INVERSE	CUFFT_INVERSE

/**
 * Executes the 3D complex-to-complex FFT plan in-place.
 */
void gpuc2cplan_exec(gpuc2cplan* plan,	///< the plan to be executed
		     float* data,	///< data to be transformed in-place
		     int direction	/// FORWARD or INVERSE
		     );

		     
/**
 * Frees the FFT plan
 * @todo not fully implemented
 */
void delete_gpuc2cplan(gpuc2cplan* plan	///< the plan to be deleted
		      );

		      
//_________________________________________________________________________________________ convolution

/**
 * A very simple and unoptimized convolution plan on the GPU
 */
typedef struct{
  
  int* size;			///< 3D size of the magnetization field
  int N;			///< total number of magnetization vectors for linear access
  
  int* paddedSize;		///< 3D size of the zero-padded magnetization buffer
  int paddedN;			///< total number of magnetization vectors in the padded magnetization buffer, for linear access
  
  
  int* paddedComplexSize;	///< 3D size of the zero-padded magnetization buffer, in complex-number format
  int paddedComplexN;		///< total number of magnetization vectors in the padded magnetization buffer in complex-number format, for linear access
  
  int len_m;			///< total number of floats in the magnetization array
  float** m_comp;		///< pointers to X, Y and Z components of magnetization, they will point into the m array passed to gpuconv1_exec() 
  int len_m_comp;		///< total number of floats in each of the m_comp array (1/3 of len_m)
  float* ft_m_i;		///< buffer for one componet of m, zero-padded and in complex-format 
  int len_ft_m_i;		///< total number of floats in ft_m_i
  
  float*** ft_kernel;		///< ft_kernel[s][d] gives the d-component of the field of a a unit vector along the s direction (in Fourier space). These components are themselves 3D fields of size paddedComplexSize. 
  int len_ft_kernel;
  int len_ft_kernel_ij;
  int len_kernel_ij;
  
  //float* h;
  int len_h;
  float** h_comp;		///< pointers to X, Y and Z components of the magnetic field, they will point into the h array passed to gpuconv1_exec() @todo should be removed as it gets only initialized after calling gpuconv1_exec(). This is confusing...
  int len_h_comp;
  float* ft_h;			///< buffer for the FFT'ed magnetic field
  int len_ft_h;
  float** ft_h_comp;		///< points to X, Y and Z components of ft_h
  int len_ft_h_comp;
  
  gpuc2cplan* fftplan;
  
}gpuconv1;

/**
 * New convolution plan.
 * 
 */
gpuconv1* new_gpuconv1(int N0,		///< X size of the magnetization vector field
		       int N1,		///< Y size of the magnetization vector field
		       int N2,  	///< Z size of the magnetization vector field
		       tensor* kernel	///< convolution kernel of size 3 x 3 x 2*N0 x 2*N1 x 2*N2
		       );

/**
 * Executes the convolution plan: convolves the source data with the stored kernel and stores the result in the destination pointer.
 * @todo: rename: execute 
 */
void gpuconv1_exec(gpuconv1* plan,	///< the plan to execute 
		   float* source, 	///< the input vector field (magnetization)
		   float* dest	///< the destination vector field (magnetic field) to store the result in
		   );

/**
 * Loads a kernel. Automatically called during new_gpuconv1(), but could be used to change the kernel afterwards.
 * @see new_gpuconv1
 */
void gpuconv1_loadkernel(gpuconv1* plan,	///< plan to load the kernel into
			 tensor* kernel		///< kernel to load (should match the plan size)
			 );

/**
 * Copies a real array to an array of complex numbers (of twice the size) and interleaves the elements with zero's (imaginary parts).
 */
void memcpy_r2c(float* source, float* dest, int nReal);

/**
 * Copies a field of real numbers into a zero-padded array and in the meanwhile converts them to complex format. Runs on the GPU.
 * @see gpu_copy_unpad_c2r
 */
void gpu_copy_pad_r2c(float* source, 	///< real input data, length = N0*N1*N2
		      float* dest,	///< complex data destination, length = 2*N0 * 2*N1 * 2*2*N2 
		      int N0,		///< X size of the real data
		      int N1,		///< Y size of the real data
		      int N2		///< Z size of the real data
		      );

/**
 * Copies the "region of interest" from a zero-padded array of complex numbers into a smaller array of real numbers. Drops the imaginary parts on the fly. Runs on the GPU.
 * @see gpu_copy_pad_r2c
 */
void gpu_copy_unpad_c2r(float* source,	///< complex input data, length = 2*N0 * 2*N1 * 2*2*N2 
			float* dest,	///< real data destination, length = N0*N1*N2
			int N0,		///< X size of the real data
			int N1,		///< X size of the real data
			int N2		///< X size of the real data
			);

/**
 * Pointwise multiplication of arrays of complex numbers. ft_h_comp_j += ft_m_i * ft_kernel_comp_ij. Runs on the GPU.
 * @todo make use of symmetry
 * @note DO NOT store in texture memory! This would be a bit faster on older devices, but actually slower on Fermi cards!
 */
void gpu_kernel_mul(float* ft_m_i,	///< multiplication input 1
		    float* ft_kernel_comp_ij, ///< multiplication input 2
		    float* ft_h_comp_j,	///< multiplication result gets added to this array
		    int nRealNumbers	///< the number of floats(!) in each of the arrays, thus twice the number of complex's in them.
		    );

		    

#ifdef __cplusplus
}
#endif
#endif