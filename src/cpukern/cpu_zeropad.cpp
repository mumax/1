#include "cpu_zeropad.h"
#include <assert.h>
#include <string.h>
#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @internal Does padding and unpadding, not necessarily by a factor 2


//begin cpu_copy_pad ------------------------
typedef struct{
  float *source, *dest;
  int S0, S1, S2, D0, D1, D2;
} cpu_copy_pad_arg;

void cpu_copy_pad_3D_t(int id){

  cpu_copy_pad_arg *arg = (cpu_copy_pad_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->S0);

  for(int i=start; i<stop; i++)
    for(int j=0; j<arg->S1; j++)
      memcpy(arg->dest + (i*arg->D1 + j)*arg->D2, arg->source + (i*arg->S1 + j)*arg->S2, arg->S2*sizeof(float));


  return;
}

void cpu_copy_pad_2D_t(int id){

  cpu_copy_pad_arg *arg = (cpu_copy_pad_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->S1);

  for(int j=start; j<stop; j++)
    memcpy(arg->dest + j*arg->D2, arg->source + j*arg->S2, arg->S2*sizeof(float));

  return;
}

void cpu_copy_pad(float* source, float* dest,
                         int S0, int S1, int S2,
                         int D0, int D1, int D2){

  assert(S0 <= D0 && S1 <= D1 && S2 <= D2);

  cpu_copy_pad_arg args;
  args.source = source;
  args.dest = dest;
  args.S0 = S0;
  args.S1 = S1;
  args.S2 = S2;
  args.D0 = D0;
  args.D1 = D1;
  args.D2 = D2;

  func_arg = (void *) (&args);

  if (S0>1)
    thread_Wrapper(cpu_copy_pad_3D_t);
  else
    thread_Wrapper(cpu_copy_pad_2D_t);

  return;
}
//end cpu_copy_pad --------------------------


//begin cpu_copy_unpad ----------------------
typedef struct{
  float *source, *dest;
  int S0, S1, S2, D0, D1, D2;
} cpu_copy_unpad_arg;

void cpu_copy_unpad_3D_t(int id){

  cpu_copy_unpad_arg *arg = (cpu_copy_unpad_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->D0);

  for(int i=start; i<stop; i++)
    for(int j=0; j<arg->D1; j++){
      memcpy(arg->dest + (i*arg->D1 + j)*arg->D2, arg->source + (i*arg->S1 + j)*arg->S2, arg->D2*sizeof(float));
    }

  return;
}

void cpu_copy_unpad_2D_t(int id){

  cpu_copy_unpad_arg *arg = (cpu_copy_unpad_arg *) func_arg;
 
  int start, stop;
  init_start_stop (&start, &stop, id, arg->D1);

  for(int j=start; j<stop; j++)
    memcpy(arg->dest + j*arg->D2, arg->source + j*arg->S2, arg->D2*sizeof(float));

  return;
}

void cpu_copy_unpad(float* source, float* dest,
                    int S0, int S1, int S2,
                    int D0, int D1, int D2){

  assert(S0 >= D0 && S1 >= D1 && S2 >= D2);

  cpu_copy_unpad_arg args;
  args.source = source;
  args.dest = dest;
  args.S0 = S0;
  args.S1 = S1;
  args.S2 = S2;
  args.D0 = D0;
  args.D1 = D1;
  args.D2 = D2;

  func_arg = (void *) (&args);

  if (D0>1)
    thread_Wrapper(cpu_copy_unpad_3D_t);
  else
    thread_Wrapper(cpu_copy_unpad_2D_t);
    

  return;
}
//end cpu_copy_unpad ------------------------




#ifdef __cplusplus
}
#endif
