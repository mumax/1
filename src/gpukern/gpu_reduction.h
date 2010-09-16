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

float gpu_sum(float* data, int size);

void gpu_sum_reduce(float* input, float* output, int blocks, int threadsPerBlock, int N);

#ifdef __cplusplus
}
#endif
#endif
