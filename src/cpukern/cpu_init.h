/**
 * @file
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef cpu_init_h
#define cpu_init_h

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Does the necessary initialization before the CPU backend can be used
 */
void cpu_init();

#ifdef __cplusplus
}
#endif
#endif
