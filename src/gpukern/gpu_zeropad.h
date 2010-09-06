/**
 * @file
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef gpu_zeropad_h
#define gpu_zeropad_h

#ifdef __cplusplus
extern "C" {
#endif


void gpu_copy_pad(float* source, float* dest,
                         int S0, int S1, int S2,        ///< source size
                         int D0, int D1, int D2         ///< dest size
                         );      


void gpu_copy_unpad(float* source, float* dest,
                         int S0, int S1, int S2,        ///< source size
                         int D0, int D1, int D2         ///< dest size
                         );


#ifdef __cplusplus
}
#endif
#endif
