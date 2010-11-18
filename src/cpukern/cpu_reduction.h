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


float cpu_reduce(int operation, float* input, float* output, float* buffer, int blocks, int threadsPerBlock, int N);

#ifdef __cplusplus
}
#endif
#endif
