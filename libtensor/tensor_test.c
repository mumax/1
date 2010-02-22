/**
 * This program runs the Unittests for libtensor and illustrates its functions.
 */

#include "libtensor.h"
#include <assert.h>

int main(int argc, char** argv){
    tensor* a;
    
    // make a 3x4 matrix
    a = new_tensor(2,  3, 4);
    
    // this is how you get the rank:
    assert(a->rank == 2);
    
    // and size
    assert(a->size[0] == 3);
    assert(a->size[1] == 4);
    
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
    
    // I/O example:
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
    return 0;
}