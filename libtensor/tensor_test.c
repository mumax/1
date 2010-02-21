/**
 * This program runs the Unittests for libtensor and illustrates its functions.
 */

#include "libtensor.h"
#include <assert.h>

int main(int argc, char** argv){
    tensor* a, *b, *c, *d;
    int i,j;
    
    // make a 3x4 matrix
    a = new_tensor(2,  3, 4);
    
    // this is how you get the rank:
    assert(a->rank == 2);
    
    // and size
    assert(a->size[0] == 3);
    assert(a->size[1] == 4);
    
    int buffer[10];
    // the data is initialized to zero:
    for(i=0; i < a->size[0]; i++){
      for(j=0; j < a->size[1]; j++){
	buffer[0] = i;
	buffer[1] = j;
	assert(*tensor_elem(a, buffer) == 0.);
      }
    }
    
    // let's set some elements...
    for(i=0; i < a->size[0]; i++){
      for(j=0; j < a->size[1]; j++){
	buffer[0] = i;
	buffer[1] = j;
	*tensor_elem(a, buffer) = i + 10. *j;
      }
    }
    
    // ... and get them back
    for(i=0; i < a->size[0]; i++){
      for(j=0; j < a->size[1]; j++){
	buffer[0] = i;
	buffer[1] = j;
	assert(*tensor_elem(a, buffer) == i + 10. *j);
      }
    }
    
    // I/O example:
    print_tensor(a, stdout);
    
    delete_tensor(a);
    
    return 0;
}