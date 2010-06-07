/**
 * @file
 * Tests for smart zero-padded FFT on the GPU
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */

#include "gputil.h"
#include "gpufft.h"
#include "tensor.h"
#include "assert.h"

int main(int argc, char** argv){
  
  print_device_properties(stderr);
  
  int N0 = 10;
  int N1 = 16;
  int N2 = 4;
  
  // make fft plan
  int* zero_pad = new int[3];
  zero_pad[X] = 1;
  zero_pad[Y] = 1;
  zero_pad[Z] = 1;
  gpu_plan3d_real_input* plan = new_gpu_plan3d_real_input(N0, N1, N2, zero_pad);

	
// make some host data and initialize _________________________________
  tensor* Host_in = new_tensor(3, N0, N1, N2);
  int N = tensor_length(Host_in);

  float*** in = tensor_array3D(Host_in);
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
				in[i][j][k] = i + j*0.01 + k*0.00001;
      }
  fprintf(stderr, "original:\n");
  format_tensor(Host_in, stderr);
// _____________________________________________________________________

	
// copy host data in zero-padded tensor ________________________________
  tensor* Host_padded_in = new_tensor(3, plan->paddedStorageSize[X], plan->paddedStorageSize[Y], plan->paddedStorageSize[Z]);
  int N_padded = tensor_length(Host_padded_in);

  float*** padded_in = tensor_array3D(Host_padded_in);
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
				padded_in[i][j][k] = in[i][j][k];
      }

	fprintf(stderr, "original, padded:\n");
  format_tensor(Host_padded_in, stderr);
// _____________________________________________________________________
  
// make data on device__________________________________________________
  tensor* Dev_in = as_tensor(new_gpu_array(N_padded), 3, plan->paddedStorageSize[X], plan->paddedStorageSize[Y], plan->paddedStorageSize[Z]);
  memcpy_to_gpu(Host_padded_in->list, Dev_in->list, N_padded);
// _____________________________________________________________________
       

	tensor* Host_padded_out = new_tensor(3, plan->paddedStorageSize[X], plan->paddedStorageSize[Y], plan->paddedStorageSize[Z]);

// test forward ________________________________________________________
  gpu_plan3d_real_input_forward(plan, Dev_in->list);
  memcpy_from_gpu(Dev_in->list, Host_padded_out->list, N_padded);
  fprintf(stderr, "\n\nforward:\n");
  format_tensor(Host_padded_out, stderr);
// _____________________________________________________________________

	
// test inverse ________________________________________________________
  gpu_plan3d_real_input_inverse(plan, Dev_in->list);
  memcpy_from_gpu(Dev_in->list, Host_padded_out->list, N_padded);
  fprintf(stderr, "\n\ninverse:\n");
  format_tensor(Host_padded_out, stderr);
// _____________________________________________________________________


// copy result to unpadded tensor ______________________________________	
  tensor* Host_out = new_tensor(3, N0, N1, N2);
  float*** padded_out = tensor_array3D(Host_padded_out);

  float*** out = tensor_array3D(Host_out);
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
				out[i][j][k] = padded_out[i][j][k]/(float)plan->paddedN;
      }
  fprintf(stderr, "Output:\n");
  format_tensor(Host_out, stderr);
// _____________________________________________________________________

// compare input <-> output after forward and inverse transform ________
  int error = 0;
  
  for(int i=0; i<N0; i++)
    for(int j=0; j<N1; j++)
      for(int k=0; k<N2; k++){
				if ( (in[i][j][k] - out[i][j][k]) > 1e-5){
					fprintf(stderr, "error element: %d, %d, %d\n", i, j, k );
                    error = 1;
                }
      }
// _____________________________________________________________________

	if(error == 0)
      fprintf(stderr, "PASS\n");
    else
      fprintf(stderr, "FAIL\n");

	return error;
}