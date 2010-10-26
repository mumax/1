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

/// Overwrites h with deltaM(m, h)
void gpu_spintorque_deltaM(float* m,       ///< magnetization (all 3 components, contiguously)
                           float* h,       ///< effective field, to be overwritten by torque
                           float alpha,    ///< damping constant
                           float beta,
                           float* u,       /// 0.5 * U_spintorque / cellsize[i]
                           float dt_gilb,  ///< dt * gilbert factor
                           int N0,         ///< length of each of the components of m, h (1/3 of m's total length)
                           int N1,
                           int N2);

#ifdef __cplusplus
}
#endif
#endif
