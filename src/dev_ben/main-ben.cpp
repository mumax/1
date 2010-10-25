#include "main-ben.h"
#include "timer.h"

param* read_param();
void initialize_m(tensor *, param *);


int main(int argc, char** argv){

 
  printf("*** Device properties ***\n");
//   print_device_properties(stdout);
  param* p = read_param();
  param_print(stdout, p);

    // initialization of kernel
  tensor* kernel = new_kernel(p);
    // initialization of m
  int* size4d = tensor_size4D(p->size);
  tensor *m = new_gputensor(4, size4d);
  initialize_m (m, p);
    // initialization of convolution
  conv_data *conv = new_conv_data(p, kernel);
    // initialization of the field evaluation plan
  fieldplan* field = new_fieldplan(p, conv);
    // initialization of the used time step function
  timestepper* ts = new_timestepper(p, field);  // allocates space for h internally

  double totalTime = 0.;
  char* fname = new char[1000];
  int* size4D = tensor_size4D(p->size);
  tensor* Host = new_tensorN(4, size4D);
  
//   evaluate_field(ts->field, m, ts->h);
//   
//   int* size4d_h = tensor_size4D(p->size);
//   tensor* hHost = new_tensorN(4, size4d_h);
//   FILE *temp_h = fopen("temp_h", "w");
//   tensor_copy_from_gpu(ts->h, hHost);
//   format_tensor(hHost, temp_h);
//   fclose(temp_h);
//   delete_tensor (hHost);


//    return(0);
//   FILE *av =fopen("./Data/m_av_fw_2e-1", "w");
  for(int i=0; i<10; i++){

    
//     tensor_copy_from_gpu(m, Host);
//     sprintf(fname, "./Data/m_%010d.t", i);
//     write_tensor_fname(Host, fname);
// 
//     tensor_copy_from_gpu(ts->h, Host);
//     sprintf(fname, "./Data/h_%010d.t", i);
//     write_tensor_fname(Host, fname);

    tensor_copy_from_gpu(m, Host);
    float mx = 0.0f;
    float my = 0.0f;
    float mz = 0.0f;
    for(int j=0; j<m->len/3; j++){
      mx += Host->list[0*m->len/3 + j];
      my += Host->list[1*m->len/3 + j];
      mz += Host->list[2*m->len/3 + j];
    }
    printf("total steps: %d\n", ts->totalSteps);
    printf("%e\t%f\t%f\t%f\n", totalTime*unittime(p), mx/(float)(m->len/3), my/(float)(m->len/3), mz/(float)(m->len/3));
//     fprintf(av, "%e\t%f\t%f\t%f\t%f\n", totalTime*unittime(p), mx/(float)(m->len/3), my/(float)(m->len/3), mz/(float)(m->len/3), (mx*mx+my*my+mz*mz)/(float)(m->len/3*m->len/3));
//     printf("\n%f\t%f\t%f\t%f\n", mx/(float)(m->len/3), my/(float)(m->len/3), mz/(float)(m->len/3), (mx*mx+my*my+mz*mz)/(float)(m->len/3*m->len/3));
    
    for(int j=0; j<100; j++)
      timestep(ts, m, &totalTime);

  }

  timer_printdetail();
//   fclose(av);

  return 0;
}


//   int* size4d_m = tensor_size4D(p->size);
//   tensor* mHost = new_tensorN(4, size4d_m);
//   FILE *temp_m = fopen("temp_m", "w");
//   tensor_copy_from_gpu(m, mHost);
//   format_tensor(mHost, temp_m);
//   fclose(temp_m);
//   delete_tensor (mHost);
// 
//   int* size4d_h = tensor_size4D(p->size);
//   tensor* hHost = new_tensorN(4, size4d_h);
//   FILE *temp_h = fopen("temp_h", "w");
//   tensor_copy_from_gpu(ts->h, hHost);
//   format_tensor(hHost, temp_h);
//   fclose(temp_h);
//   delete_tensor (hHost);

//     int* size4d_m = tensor_size4D(p->size);
//     tensor* mHost = new_tensorN(4, size4d_m);
//     FILE *temp_m = fopen("temp_m", "w");
//     tensor_copy_from_gpu(m, mHost);
//     format_tensor(mHost, temp_m);
//     fclose(temp_m);
//     delete_tensor (mHost);

//   tensor* gHost = new_tensorN(2, kernel->size);
//   FILE *temp_g = fopen("temp_g", "w");
//   tensor_copy_from_gpu(kernel, gHost);
//   format_tensor(gHost, temp_g);
//   fclose(temp_g);
//   delete_tensor (gHost);




param* read_param(){
  
  param* p = new_param();

  p->msat = 800E3;
  p->aexch = 1.3E-11;
  p->alpha = 1.0;

  p->size[X] = 1;
  p->size[Y] = 32;
  p->size[Z] = 128;

//   p->size[X] = 2;
//   p->size[Y] = 4;
//   p->size[Z] = 8;

  double L = unitlength(p);
  printf("unitlength: %e\n", L);
  
  p->cellSize[X] = 3.0E-9 / L;
//   p->cellSize[X] = 3.0E-9 / L;
  p->cellSize[Y] = 3.90625E-9 / L;
  p->cellSize[Z] = 3.90625E-9 / L;

  p->demagCoarse[X] = 1;
  p->demagCoarse[Y] = 1;
  p->demagCoarse[Z] = 1;
  
  p->demagPeriodic[X] = 0;
  p->demagPeriodic[Y] = 0;
  p->demagPeriodic[Z] = 0;

  int zero_pad[3];
  for (int i=0; i<3; i++){
    zero_pad[i] = (!p->demagPeriodic[i]) ? 1:0;
    p->kernelSize[i] = (1 + zero_pad[i]) * p->size[i]/p->demagCoarse[i]; 
  }
  if (p->size[X]==1) 
    p->kernelSize[X] = 1;

  p->kernelType = KERNEL_MICROMAG3D;
//   p->kernelType = KERNEL_MICROMAG2D;
  
//  p->solverType = SOLVER_ANAL_FW;
   p->solverType = SOLVER_ANAL_PC;
//   p->solverType = SOLVER_HEUN;

  p->exchType = EXCH_6NGBR;

//Depending on the kerneltype and/or a coarse grid evaluation of the demag field, some/all components of the exchange fields need to be added classically
//   exchInConv[comp] = -1 : for component 'comp' of the field, exchange is not computed (not classically, nor in convolution)
//   exchInConv[comp] =  0 : for component 'comp' of the field, exchange is computed classically 
//   exchInConv[comp] =  1 : for component 'comp' of the field, exchange is included in the convolution

  if (p->exchType>0 && p->demagCoarse[X]==1 && p->demagCoarse[Y]==1 && p->demagCoarse[Z]==1){
    switch (p->kernelType){
      case KERNEL_MICROMAG3D:
        p->exchInConv[X] = 1;
        p->exchInConv[Y] = 1;
        p->exchInConv[Z] = 1;
        break;
      case KERNEL_MICROMAG2D:
        p->exchInConv[X] = 0;
        p->exchInConv[Y] = 1;
        p->exchInConv[Z] = 1;
        break;
      default:
        fprintf(stderr, "abort: no valid kernelType\n");
        abort();
    }
  }
  else{
    p->exchInConv[X] = 0;
    p->exchInConv[Y] = 0;
    p->exchInConv[Z] = 0;
  }
//**********************************************************************************************************************

  double T = unittime(p);
  p->maxDt = 0.2E-12 / T;
 
  return p;

}



void initialize_m(tensor *m, param *p){

  int* size4D = tensor_size4D(p->size);
  tensor* mHost = new_tensorN(4, size4D);
  for(int i=0; i<mHost->len; i++)
    mHost->list[i] = 0.57735026;

//   for(int i=0*mHost->len/3; i<3*mHost->len/3; i++)
//     mHost->list[i] = 1.0;

  tensor_copy_to_gpu(mHost, m);
  
  delete_tensor(mHost);
  
  return;
}
