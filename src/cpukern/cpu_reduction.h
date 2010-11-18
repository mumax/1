/**
 * @file
 *
 * @author Arne Vansteenkiste
 */
#ifndef cpu_reduction_h
#define cpu_reduction_h

#ifdef __cplusplus
extern "C" {
#endif

#define REDUCE_ADD 1
#define REDUCE_MAX 2
#define REDUCE_MAXABS 3
#define REDUCE_MIN 4


/// Reduces the input (array on device)
float cpu_reduce(int operation,     ///< REDUCE_ADD, REDUCE_MAX, ... 
                 float* input,      ///< input data on device
                 float* devbuffer,  ///< device buffer of size "blocks"
                 float* hostbuffer, ///< host buffer of size "blocks"
                 int blocks,        ///< blocks * threadsPerBlock * 2 = N
                 int threadsPerBlock,///< threads per thread block (maximum is device dependent)
                 int N              ///< input size
                 );

#ifdef __cplusplus
}
#endif
#endif
