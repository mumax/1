/**
 * @file
 *
 * @author Ben Van de Wiele
 */
#ifndef gpu_anal_h
#define gpu_anal_h

#ifdef __cplusplus
extern "C" {
#endif

void gpu_anal_fw_step_unsafe(float* m, float* h, float dt, float alpha, int N);

#ifdef __cplusplus
}
#endif
#endif
