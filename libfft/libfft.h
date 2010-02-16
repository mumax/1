/*
 * The aim of this file is to provide a common interface for different FFT libraries.
 * E.g.: we may run FFT's on CPU's, FFT over MPI, or an FFT on GPU's. This common
 * interface hides the differences between those implementations: independent of
 * the underlying library we can just write: fft_execute(plan)
 * instead of fftwf_execute(fftwf_plan) / cufft_execute(cufft_plan) / mpifft_execute(mpi_plan) / ...
 * 
 * Thus for each library, we need to write a small file that wraps the functions
 * with those provided by this interface, effectively hiding the original function names
 * and other details. E.g., libfft_cpu.c wraps a simple FFTW3 implementation. 
 * We can use it by just linking against libfft-cpu.so.  Later, we can replace it by an
 * other object file which runs the fft on GPU's or so. The main program won't notice
 * the difference since all libraries are all wrapped in the same interface.
 * 
 * In the spirit of the FFTW design, all functions should be stateless. 
 * If they need to access some state, like an FFTW plan, it gets passed
 * as a void* parameter. This state gets initialized by XXX_init() methods, who return the state as a void*.
 * E.g.: fft_init(N0, N1, N2) may create an FFTW plan and return it. The plan is to be passed to each call of
 * fft_forward(plan, data).
 *
 * This is important as we need some flexibilty:
 * there may need to be more than one FFT function during a simulation. E.g., for
 * two coupled systems who each need their own demag field. A typical C++ approach might be to implement a
 * class FFT_Transformer, and create many instances of it. However, we need plain C because we have to
 * interface with low-level languages like NVIDIA CUDA. Therefore, the state passed as a void* plays
 * the role of the FFT_Transformer class: many "instances" may be created during a simulation, and they
 * don't interfere with each other. The price we pay is that we need to pass the state explicitly.
 *
 * Only the low-level, high-performance functions like FFT's, convolutions, ... are accessed through
 * the C-interface. Higher-level constructs, like a simulation control flow, can be implemented in
 * any language that links with C
 *
 * NOTE on style. As this code will be seen by simple compilers like nvcc, cgo, ..., the style needs
 * to be as standard as possible. E.g.:
 * void function(void);
 *
 */
#ifndef LIBSIM_H
#define LIBSIM_H

#include <stdio.h>

// allow inclusion in C++ code
#ifdef __cplusplus
extern "C" {
#endif


/* Define SINGLE_PRECISSION or DOUBLE_PRECISSION. Can be overridden via a command-line flag. */
#ifndef SINGLE_PRECISSION
#ifndef DOUBLE_PRECISSION

#define SINGLE_PRECISSION
//#define DOUBLE_PRECISSION
#endif
#endif


/**
 * We will probably do everything with single-precission floating point data. Nevertheless we use 'real' instead of float so that later we might still re-define it as double. 
 */
#ifdef SINGLE_PRECISSION
typedef float real;
//#define real float
#endif

#ifdef DOUBLE_PRECISSION
typedef double real;
//#define real double
#endif

/**
 * Complex numbers are represented by fftw, cufft, ... as real[2], where element 0 represents the real part and element 1 the imaginary part. This type is binary compatible with fftw(f)_complex, complex<real>, ...
 * We do not explicitly use this type, just arrays of reals containing interleaved real and imaginary parts.
 */
typedef real complex_t[2];



/** Returns a number to identify the library (used mainly for debugging, to be sure we have linked against the expected library) */
int fft_version(void);

/** Should be called once to initialize the library. Implementations using MPI may need this. Todo: the command-line arguments should be passed */
void fft_init(void);

/** Should be called once to finalize the library. Implementations using MPI may need this. */
void fft_finalize(void);

/**
 * fft_init functions get called once to initialize the FFT for transformation of an N0 x N1 x N2 array. 
 * They return a plan (or any other state needed by a specific FFT implementation).
 *
 * NOTE: Depending on the underlying library, fft_init_XXX may destroy the data in
 * the source and dest arrays. They should thus be initialized after fft_init_XXX.
 *
 * NOTE: The implementation must also accept a two-dimensional array of size N0 x N1 x 1.
 * This is important because many micromagnetic simulations of thin-films are 2D.
 */

/** real to complex forward transform */
void* fft_init_forward(int N0, int N1, int N2, real* source, real* dest);

/** complex to real backward transform */
void* fft_init_backward(int N0, int N1, int N2, real* source, real* dest);

/** Gets called once to free the FFT resources */
void fft_destroy_plan(void* plan);

/** Performs an unnormalized FFT. The result is multiplied by a factor sqrt(N), with N the total number of points, compared to a normalized FFT. */
void fft_execute(void* plan);


/**
 * Allocates memory to store an N0 x N1 x N2 array of complex numbers.
 * Different FFT implementations may provide optimized methods here.
 * E.g.: FFTW double-word aligns the data for the sake of SIMD instructions,
 * CUFFT may allocate in page-locked memory which has a higher throughput,
 * data may be allocated in GPU memory, ...
 */
real* fft_malloc(int N0, int N1, int N2);

/** Frees memory allocated by fft_malloc */
void fft_free(void* data);


#ifdef __cplusplus
}
#endif

#endif