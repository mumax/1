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

//void cpu_anal_fw_step(float* m, float* h, float dt, float alpha, int N);
void cpu_anal_fw_step(float dt, float alpha, int N, float *min, float *mout, float *h);

#ifdef __cplusplus
}
#endif
#endif
