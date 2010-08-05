/**
 * @file
 *
 * @author Ben Van de Wiele
 * @author Arne Vansteenkiste
 */
#ifndef cpu_anal_h
#define cpu_anal_h

#ifdef __cplusplus
extern "C" {
#endif

void cpu_anal_fw_step_unsafe(float* m, float* h, float dt, float alpha, int N);

#ifdef __cplusplus
}
#endif
#endif
