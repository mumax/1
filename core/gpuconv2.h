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
  
}gpur2cplan;

/**
 * There is a difficulty with real-to-complex FFT's:
 * the last dimension must be made 2 complex numbers larger,
 * but then it does not fit the stride anymore.
 * Extra padding? Out-of-place transform?
 */
gpur2cplan* new_gpur2cplan(int N0,	///< size in x-direction
			   int N1,	///< size in y-direction
			   int N2	///< size in z-direction
			   );

/**
 * Executes in-place.
 */
void gpur2cplan_forward(gpur2cplan* plan,	///< the plan to be executed
		        float* data	///< data to be transformed in-place
			);

/**
 * Executes in-place.
 */
void gpur2cplan_backward(gpur2cplan* plan,	///< the plan to be executed
		        float* data	///< data to be transformed in-place
			);

		     
/**
 * Frees the FFT plan
 */
void delete_gpur2cplan(gpur2cplan* plan	///< the plan to be deleted
		      );

		      
//_________________________________________________________________________________________ convolution

/**
 * 
 */
typedef struct{
  
 /* int* size;			///< 3D size of the magnetization field
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
  float** h_comp;		///< pointers to X, Y and Z components of the magnetic field, they will point into the h array passed to gpuconv1_exec() 
  int len_h_comp;
  float* ft_h;			///< buffer for the FFT'ed magnetic field
  int len_ft_h;
  float** ft_h_comp;		///< points to X, Y and Z components of ft_h
  int len_ft_h_comp;
  
  gpur2cplan* fftplan;
 */ 
}gpuconv2;

/**
 * New convolution plan.
 * 
 */
gpuconv2* new_gpuconv2(int N0,		///< X size of the magnetization vector field
		       int N1,		///< Y size of the magnetization vector field
		       int N2,  	///< Z size of the magnetization vector field
		       tensor* kernel	///< convolution kernel of size 3 x 3 x 2*N0 x 2*N1 x 2*N2
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