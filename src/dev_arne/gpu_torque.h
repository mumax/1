/**
 * @file
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
//<<<<<<< HEAD:src/dev_arne/gpu_torque.h
#ifndef gpu_torque_h
#define gpu_torque_h
//=======
//#ifndef GPUTORQUE_H
//#define GPUTORQUE_H

//#include "gpukern.h"
//#include "timer.h"
//>>>>>>> arne:src/core/gputorque.h

#ifdef __cplusplus
extern "C" {
#endif



/// Overwrites h with deltaM(m, h)
void gpu_deltaM(float* m,       ///< magnetization (all 3 components, contiguously)
                float* h,       ///< effective field, to be overwritten by torque
                float alpha,    ///< damping constant
                float dt_gilb,  ///< dt * gilbert factor
                int N           ///< length of each of the components of m, h (1/3 of m's total length)
                );

#ifdef __cplusplus
}
#endif
#endif
