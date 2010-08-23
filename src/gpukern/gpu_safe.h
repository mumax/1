/**
 * @file
 *
 * gpu_safe() should be wrapped around cuda functions to check for a non-zero error status.
 *
 * @author Arne Vansteenkiste
 */
#ifndef gpu_safe_h
#define gpu_safe_h

#ifdef __cplusplus
extern "C" {
#endif


/**
 * This function should be wrapped around cuda functions to check for a non-zero error status.
 * It will print an error message and abort when neccesary.
 * @code
 * gpu_safe( cudaMalloc(...) );
 * @endcode
 */
// void gpu_safe(int status    ///< CUDA return status
//           );

#define gpu_safe(s) assert(s == 0);
          

#ifdef __cplusplus
}
#endif
#endif
