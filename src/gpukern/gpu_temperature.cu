/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

///@author Arne Vansteenkiste

#include "gpu_temperature.h"
#include "gpu_safe.h"
#include <curand.h>


#ifdef __cplusplus
extern "C" {
#endif

#define CUDA_CALL(x) do { if((x) != cudaSuccess) { \
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x) != CURAND_STATUS_SUCCESS) { \
printf("Error at %s:%d\n",__FILE__,__LINE__);\
return EXIT_FAILURE;}} while(0)


curandGenerator_t thermal_randgen = NULL;
long long int thermal_randgen_seed = 0;

void gpu_gaussian_noise(float* devData, float mean, float stddev, int n) {
	if(thermal_randgen == NULL){
		assert(curandCreateGenerator(&thermal_randgen, CURAND_RNG_PSEUDO_DEFAULT) == 0);
		assert(curandSetPseudoRandomGeneratorSeed(thermal_randgen, thermal_randgen_seed) == 0);
	}
	assert(curandGenerateNormal(thermal_randgen, devData, n, mean, stddev) == 0);
	gpu_sync();
}


#ifdef __cplusplus
}
#endif
