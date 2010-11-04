/**
 * @file
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef gpu_reduction_h
#define gpu_reduction_h

#ifdef __cplusplus
extern "C" {
#endif

#define REDUCE_ADD 1
#define REDUCE_MAX 2
#define REDUCE_MAXABS 3
#define REDUCE_MIN 4


float gpu_reduce(int operation, float* input, float* output, float* buffer, int blocks, int threadsPerBlock, int N);

float gpu_sum(float* data, int size);
  
/// @internal Low-level partial sum function
void gpu_partial_sums(float* input,        ///< input data. size = N @todo check if it gets overwritten
                    float* output,       ///< partially reduced data, usually reduced further on CPU. size = blocks
                    int blocks,          ///< patially reduce in X blocks, partial results in output. blocks = divUp(N, threadsPerBlock*2)
                    int threadsPerBlock, ///< use X threads per block: @warning must be < N
                    int N                ///< size of input data, must be > threadsPerBlock
                    );

#ifdef __cplusplus
}
#endif
#endif
