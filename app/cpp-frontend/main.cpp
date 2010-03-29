#include "main.h"
#include "../../core/tensor.h"
#include "../../core/conv_gpu.h"
#include "../../core/euler.h"
#include <assert.h>
#include <stdio.h>

int main(int argc, char** argv){
  
    assert(argc == 3);
    
    FILE* mfile = fopen(argv[1], "rb");
    tensor* m = read_tensor(mfile);
    fclose(mfile);
    printf("read m: %d x %d x %d\n", m->size[1], m->size[2], m->size[3]);
    
    FILE* kernelfile = fopen(argv[2], "rb");
    tensor* kernel = read_tensor(kernelfile);
    fclose(kernelfile);
    printf("read kernel: %d x %d x %d\n", kernel->size[2], kernel->size[3], kernel->size[4]);
    
    char* buffer = (char*)calloc(200, sizeof(char));
    eulersolver* solver = new_euler(m, new_convplan(m->size[1], m->size[2], m->size[3], kernel->list), 0.01);
    for(int i=0; i<100; i++){
      printf("%d\n", i);
      for(int j=0; j<100; j++){
	euler_step(solver);
      }
      sprintf(buffer, "m%d.t", i);
      FILE* mifile = fopen(buffer, "wb");
      write_tensor(m, mifile);
      fclose(mifile);
    }
    

    
    return 0;
  
//   int N0 = 4, N1 = 4, N2 = 2;
//   int* size = new int[3];
//   size[0] = N0; size[1] = N1, size[2] = N2;
//   
//   tensor* m = new_tensor(4, 3, N0, N1, N2);
//   tensor* h = new_tensor(4, 3, N0, N1, N2);
//   
//   for(int i=0; i<tensor_length(m); i++){
//     m->list[i] = i/100.;//(rand()%1000 + 1)/1000.0;
//   }
//   
//   tensor* kernel = new_tensor(5, 3, 3, 2*N0, 2*N1, 2*N2);
//   // unit kernel:
//   for(int s=0; s<3; s++){
//     *tensor_get(kernel, 5, s, s, 0, 0, 0) = 1.0;
//   }
//   
//   convplan* plan = new_convplan(N0, N1, N2, kernel->list);
// 
//   //format_tensor(kernel, stdout);
//   printf("M\n\n");
//   format_tensor(m, stdout);
//   
//   conv_execute(plan, m->list, h->list);
//   
//   printf("H\n\n");
//   format_tensor(h, stdout);
//   
//   delete_convplan(plan);
// 
// //   float rmserror = 0.;
// //   for(int i=0; i<tensor_length(m); i++){
// //     rmserror += sqr(m->list[i] - h->list[i]);
// //     assert(m->list[i] != 0.); // it's really easy to make such a mistake that both m and h contain only zero's. 
// //   }
// //   rmserror = sqrt(rmserror);
// //   printf("unit convolution rms error = %f\n", rmserror);
// //   assert(rmserror < 1E-5);
//   
//   printf("PASS\n");
//   return 0;
}