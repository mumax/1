/**
 * @file
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef cpu_torque_h
#define cpu_torque_h

#ifdef __cplusplus
extern "C" {
#endif



/// Overwrites h with deltaM(m, h)
void cpu_deltaM(float* m,       ///< magnetization (all 3 components, contiguously)
                float* h,       ///< effective field, to be overwritten by torque
                float alpha,    ///< damping constant
                float dt_gilb,  ///< dt * gilbert factor
                int N           ///< length of each of the components of m, h (1/3 of m's total length)
                );

#ifdef __cplusplus
}
#endif
#endif
