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

float gpu_reduce(float* data, int size);

// void _gpu_reduce(int size, int threads, int blocks, float* d_idata, float* d_odata);

#ifdef __cplusplus
}
#endif
#endif
