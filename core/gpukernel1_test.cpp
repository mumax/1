/**
 * @file
 * Tests for kernel initialization
 *
 * @author Ben Van de Wiele
 */

#include "gpukernel1.h"

int main(int argc, char** argv){
  
  int N0 = 4;
  int N1 = 8;
  int N2 = 16;
  
  int* zero_pad = new int[3];
  zero_pad[X] = 1;
  zero_pad[Y] = 1;
  zero_pad[Z] = 1;

  int* repetition = new int[3];
	repetition[X] = repetition[Y] = repetition[Z] = 0;

	float* FD_cell_size = new float[3];
	FD_cell_size[X] = 1.0f;
	FD_cell_size[Y] = 1.2f;
	FD_cell_size[Z] = 1.4f;

	tensor *dev_kernel;
	gpu_init_Greens_kernel1(dev_kernel, N0, N1, N2, zero_pad, repetition, FD_cell_size);

	fprintf(stderr, "\nPASS\n");

	return 0;
}