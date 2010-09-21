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


void cpu_reduce(int operation, float* input, float* output, int blocks, int threadsPerBlock, int N);

// float cpu_sum(float* data, int size);
  
/// @internal Low-level partial sum function
// void gpu_partial_sums(float* input,        ///< input data. size = N @todo check if it gets overwritten
//                     float* output,       ///< partially reduced data, usually reduced further on CPU. size = blocks
//                     int blocks,          ///< patially reduce in X blocks, partial results in output. blocks = divUp(N, threadsPerBlock*2)
//                     int threadsPerBlock, ///< use X threads per block: @warning must be < N
//                     int N                ///< size of input data, must be > threadsPerBlock
//                     );

#ifdef __cplusplus
}
#endif
#endif
