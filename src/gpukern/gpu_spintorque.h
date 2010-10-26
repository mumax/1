/**
 * @file
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef gpu_spintorque_h
#define gpu_spintorque_h

#ifdef __cplusplus
extern "C" {
#endif

void gpu_directionial_diff(float ux, float uy, float uz, float* in, float* out, int N0, int N1, int N2);

#ifdef __cplusplus
}
#endif
#endif
