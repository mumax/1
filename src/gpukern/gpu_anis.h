/**
 * @file
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef gpu_anis_h
#define gpu_anis_h

#ifdef __cplusplus
extern "C" {
#endif

/// Adds a linear anisotropy contribution to h:
/// h_i += Sum_i k_ij * m_j
/// Used for edge corrections.
void gpu_add_lin_anisotropy(float* hx, float* hy, float* hz,
                            float* mx, float* my, float* mz,
                            float* kxx, float* kyy, float* kzz,
                            float* kyz, float* kxz, float* kxy,
                            int N);

#ifdef __cplusplus
}
#endif
#endif
