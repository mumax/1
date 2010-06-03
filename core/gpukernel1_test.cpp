/**
 * @file
 * Tests for kernel initialization
 *
 * @author Ben Van de Wiele
 */

#include "gputil.h"
#include "gpufft.h"
#include "tensor.h"
#include "gpukernel1.h"
#include "assert.h"

int main(int argc, char** argv){
  
  print_device_properties(stderr);
  
  int N0 = 2;
  int N1 = 4;
  int N2 = 8;
  
  int* zero_pad = new int[3];
  zero_pad[X] = 0;
  zero_pad[Y] = 0;
  zero_pad[Z] = 0;

  int* repetition = new int[3];
	repetion[X] = repetition[Y] = repetition[Z] = 0;

	float* FD_cell_size = new float[3];
	FD_cell_size[X] = 1.0;
	FD_cell_size[Y] = 1.0;
	FD_cell_size[Z] = 1.0;

	tensor *kernel;
	init_Greens_kernel1(kernel, N0, N1, N2, zero_pad, repetition, FD_cell_size);

	fprintf(stderr, "PASS\n");

	return 0;
}