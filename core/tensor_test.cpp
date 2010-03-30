/**
* This program runs the Unittests for "tensor" and illustrates its functions.
*/
#include "tensor.h"
#include <assert.h>

int main(int argc, char** argv){
  printf("tensor_test: ");
  tensor* a;
  
  // make a 30x40 matrix
  a = new_tensor(2,  30, 40);
  
  // this is how you get the rank:
  assert(a->rank == 2);
  
  // and size
  assert(a->size[0] == 30);
  assert(a->size[1] == 40);
  
  int buffer[10];
  // the data is initialized to zero:
  for(int i=0; i < a->size[0]; i++){
    for(int j=0; j < a->size[1]; j++){
      buffer[0] = i;
      buffer[1] = j;
      assert(*tensor_elem(a, buffer) == 0.);
    }
  }
  
  // let's set some elements...
  for(int i=0; i < a->size[0]; i++){
    for(int j=0; j < a->size[1]; j++){
      buffer[0] = i;
      buffer[1] = j;
      *tensor_elem(a, buffer) = i + 10. *j;
    }
  }
  
  // ... and get them back
  for(int i=0; i < a->size[0]; i++){
    for(int j=0; j < a->size[1]; j++){
      buffer[0] = i;
      buffer[1] = j;
      assert(*tensor_elem(a, buffer) == i + 10. *j);
    }
  }
  
  // easier way
  float** a_arr = tensor_array2D(a);
    for(int i=0; i < a->size[0]; i++){
    for(int j=0; j < a->size[1]; j++){
      assert(a_arr[i][j] == i + 10. *j);
    }
  }
  
  // I/O example:
  // print to screen
  //print_tensor(a);
  
  // write:
  FILE* out;
  out = fopen ( "iotest" , "wb" );
  write_tensor(a, out);
  fclose (out);
  
  // read back in:
  FILE* in;
  in = fopen ( "iotest" , "rb" );
  tensor* b = read_tensor(in);
  fclose (out);
  
  // check if read back equals original:
  assert(a->rank == b->rank);
  for(int i=0; i<a->rank; i++){
    assert(a->size[i] == b->size[i]);
  }
  
  for(int i=0; i < tensor_length(a); i++){
    assert(a->list[i] == b->list[i]);
  }
  
  delete_tensor(a);
  delete_tensor(b);
  
  // tensor construction when rank is not known in advance
  //int[] buffer = new int[2];
  buffer[0] = 5;
  buffer[1] = 6;
  tensor* c = new_tensorN(2, buffer);
  assert(c->rank == 2);
  assert(c->size[0] == 5);
  assert(c->size[1] == 6);
  
  // the data is initialized to zero:
  for(int i=0; i < c->size[0]; i++){
    for(int j=0; j < c->size[1]; j++){
      buffer[0] = i;
      buffer[1] = j;
      assert(*tensor_elem(c, buffer) == 0.);
    }
  }
  
  // let's set some elements...
  for(int i=0; i < c->size[0]; i++){
    for(int j=0; j < c->size[1]; j++){
      buffer[0] = i;
      buffer[1] = j;
      *tensor_elem(c, buffer) = i + 10. *j;
    }
  }
  
  // ... and get them back
  for(int i=0; i < c->size[0]; i++){
    for(int j=0; j < c->size[1]; j++){
      buffer[0] = i;
      buffer[1] = j;
      assert(*tensor_elem(c, buffer) == i + 10. *j);
    }
  }
  
  
  delete_tensor(c);
  
  // this represents a 10x20x30 field of 3-component vectors:
  tensor* d = new_tensor(4, 3, 10, 20, 30);
  
  // let's set all the X components to 1, Y to 2 and Z to 3:
  for(int c=0; c<3; c++){
    for(int i=0; i<d->size[1]; i++){
      for(int j=0; j<d->size[2]; j++){
	for(int k=0; k<d->size[3]; k++){
	  buffer[0] = c;
	  buffer[1] = i;
	  buffer[2] = j;
	  buffer[3] = k;
	  
	  *tensor_elem(d, buffer) = (float)c + 1.0;
	  
	}
      }
    }
  }
  //print_tensor(d);
  
  // we can take a tensor slice that contains only one component:
  tensor* dx = tensor_component(d, 0);
  tensor* dy = tensor_component(d, 1);
  tensor* dz = tensor_component(d, 2);
  // and we could even slice those further to get the Nth row, etc... 
  
  //print_tensor(dx);
  
  // in this case, each component is a rank 3 tensor, while its parent is rank 4:
  assert(dx->rank == 3);
  // and its size is 10x20x30:
  assert(dx->size[0] == 10);
  assert(dx->size[1] == 20);
  assert(dx->size[2] == 30);
  // it shares the data with the parent tensor, let's check that:
  for(int i=0; i<tensor_length(dx); i++){
    assert(dx->list[i] == 1.0);
    assert(dy->list[i] == 2.0);
    assert(dz->list[i] == 3.0);
  }
  
  // as parent and component tensor share their data, editing one affects the other:
  // let's set dy[1][2][3] to some value
  int* index = new int[10];
  index[0] = 1;
  index[1] = 2;
  index[2] = 3;
  *tensor_elem(dy, index) = 123.0;
  
  // now d[1][1][2][3] contains that value! 
  index[0] = 1; //1=Y
  index[1] = 1;
  index[2] = 2;
  index[3] = 3;
  assert(*tensor_elem(d, index) == 123.0);
  
  // access the tensor as an ordinary 3D array:
  float*** dy_arr = tensor_array3D(dy);
  assert(dy_arr[1][2][3] == 123.0);
  float**** d_arr = tensor_array4D(d);
  assert(d_arr[1][1][2][3] == 123.0);
  
  // components need te freed like this:
  delete_tensor_component(dx);
  delete_tensor_component(dy);
  delete_tensor_component(dz);
  // because the parent still exists and may still acces the data:
  assert(d->list[0] == 1.0); // data still allocated
  // now that the components are gone, we can safely delete the parent in the usual way:
  delete_tensor(d);
  // make sure you don't use components anymore when the parent has been deleted!
  
  //     tensor* x = new_tensor(3, 4, 5, 6);
  //     buffer[0] = 0;
  //     buffer[1] = 0;
  //     buffer[2] = 0;
  //     float z = *tensor_elem(x, buffer);
  //     format_tensor(x, stdout);
  //    
  printf("PASS\n");
  return 0;
}