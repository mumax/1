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
 * @todo 
  TODO voor convolutie (Ben)

1.  Greense functies 
	-> 	dienen gegenereerd te worden in strided formaat
	-> 	is symmetrische rank 4 tensor (vb: gxy = gyx, gxz = gzx, ..., slechts 2/3 van geheugen nodig)
	->	Enkel reeel deel in Fourier domein (halveert geheugen vereisten)
  ->  implementatie algemene Greense tensor nodig met als input te gebruiken Greense functie 
	->  Er dient rekening gehouden te worden met mogelijke periodiciteit

2.  seriele berekening veldwaarden gunstiger
	-> 	beter seriele berekening van H_x, H_y, H_z als
				a. H^FFT = g^FFT_xx* m^FFT_x + g^FFT_xy* m^FFT_y + g^FFT_xz* m^FFT_z
				b. H_x = inv_FFT(H^FFT)
				c. H^FFT = g^FFT_xy* m^FFT_x + g^FFT_yy* m^FFT_y + g^FFT_yz* m^FFT_z
				d. H_y = inv_FFT(H^FFT)
				e. H^FFT = g^FFT_xz* m^FFT_x + g^FFT_yz* m^FFT_y + g^FFT_zz* m^FFT_z
				f. H_z = inv_FFT(H^FFT)
			Op die manier enkel geheugen nodig voor H^FFT (en niet voor elke component H^FFT_x, H^FFT_y, H^FFT_z)
			Antw: Ik denk dat ik nu slechts even veel geheugen gebruik: Ik houd 3 H^FFT componenten in het geheugen,
			maar slechts één m^FFT component, jij één H^FFT maar 3 m^FFT's. Of heb ik het mis op? (Arne.)
			Opm: misschien kunnen we wel één buffer uitsparen door eerst alle m_i te FFT-en en dan een "in-place"
			kernel vermenigvuldiging te doen. Per element wordt dan m_x[i], m_y[i], m_z[i] gebufferd in
			locale variablen, daarna wordt m^FFT element per element overschreven door H^FFT...
			
	->  H^FFT dient dezelfde dimensies te hebben als andere strided FFT tensoren

3.  Transponeren matrices
	->  is versnelling mogelijk door nullen niet te transponeren?
	->  In place transponeren

4.  Omtrent de FFT routines
	-> Waarschijnlijk beter om FFT routines in een aparte bibliotheek te steken wegens mogelijk gebruik in andere convoluties
	-> implementatie 2D varianten:
				Uitbreiding van de huidige routines of aparte routines? mogelijkheden:
				a. Aparte routines voor 3D en 2D: bij aanroepen if constructies nodig (if 3D, if 2D)
				b. uitbreiding routines:
						- extra argument 2D of 3D, met daarna daarna twee totaal verschillende code blokken
						- geen extra argumenten, maar op basis van dimensies in argument.

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
#include "gpufft.h"

#ifdef __cplusplus
extern "C" {
#endif

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

   int len_m;					  ///< total number of floats in the magnetization array
   int len_m_comp;			///< total number of floats in each of the m_comp array (1/3 of len_m)
   float* ft_m_i;				///< buffer for one componet of m, zero-padded and in complex-format 
   int len_ft_m_i;			///< total number of floats in ft_m_i

   float*** ft_kernel;	///< ft_kernel[s][d] gives the d-component of the field of a a unit vector along the s direction (in Fourier space). These components are themselves 3D fields of size paddedComplexSize. 
   int len_ft_kernel;
   int len_ft_kernel_ij;
   int len_kernel_ij;
   
   int len_h;
   int len_h_comp;
   float* ft_h;			    ///< buffer for the FFT'ed magnetic field
   int len_ft_h;
   float** ft_h_comp;		///< points to X, Y and Z components of ft_h
   int len_ft_h_comp;
   
   gpu_plan3d_real_input* fftplan;
  
}gpuconv2;



/**
 * New convolution plan.
 * 
 */
gpuconv2* new_gpuconv2(int N0,						///< X size of the magnetization vector field
		       int N1,		            				///< Y size of the magnetization vector field
		       int N2,  	            				///< Z size of the magnetization vector field
		       tensor* kernel,	      				///< convolution kernel of size 3 x 3 x 2*N0 x 2*N1 x 2*N2
		       int* zero_pad          				///< 3 ints, should be 1 or 0, meaning zero-padding or no zero-padding in X,Y,Z respectively
		       );

/**
 * Executes the convolution plan: convolves the source data with the stored kernel and stores the result in the destination pointer.
 */
void gpuconv2_exec(gpuconv2* plan, 				///< the plan to execute 
		   float* source, 	           				///< the input vector field (magnetization)
		   float* dest		             				///< the destination vector field (magnetic field) to store the result in
		   );

/**
 * Loads a kernel. Automatically called during new_gpuconv2(), but could be used to change the kernel afterwards.
 * @see new_gpuconv2
 */
void gpuconv2_loadkernel(gpuconv2* plan,	///< plan to load the kernel into
			 tensor* kernel		                  ///< kernel to load (should match the plan size)
			 );

/**
 * Pointwise multiplication of arrays of complex numbers. ft_h_comp_j += ft_m_i * ft_kernel_comp_ij. Runs on the GPU.
 * Makes use of kernel symmetry
 * @note DO NOT store in texture memory! This would be a bit faster on older devices, but actually slower on Fermi cards!
 */
void gpu_kernel_mul2(float* ft_m_i,		    ///< multiplication input 1
		     float* ft_kernel_comp_ij,      	///< multiplication input 2
		     float* ft_h_comp_j, 	            ///< multiplication result gets added to this array
		     int nRealNumbers									///< the number of floats(!) in each of the arrays, thus twice the number of complex's in them.
		     );

/**
 * @internal
 * Copies 3D data to a zero-padded, strided destination. Runs on the GPU.
 */
void gpuconv2_copy_pad(gpuconv2* conv, 		///< this convolution plan contains the sizes of both arrays
		       float* source,   							///< source data on GPU, should have size: conv->size
		       float* dest										///< destination data on GPU, should have size: conv->paddedStorageSize
		       );

/**
 * @internal
 * Copies 3D data from a zero-padded and strided destination. Runs on the GPU
 */
void gpuconv2_copy_unpad(gpuconv2* conv,	///< this convolution plan contains the sizes of both arrays
			 float* source,  										///< destination data on GPU, should have size: conv->paddedStorageSize
			 float* dest	 											///< source data on GPU, should have size: conv->size
			 );

#ifdef __cplusplus
}
#endif
#endif