/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

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
