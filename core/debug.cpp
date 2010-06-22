#include "debug.h"

#ifdef __cplusplus
extern "C" {
#endif

int _debug_verbosity = 1;

void debug_verbosity(int level){
  assert(level >= 0 && level <= 3);
  _debug_verbosity = level;
}


void debug(char* message){
#ifndef NDEBUG
  if(_debug_verbosity > 0){
    fprintf(stderr, "%s\n", message);
  }
#endif
}


void debugv(char* message){
#ifndef NDEBUG
  if(_debug_verbosity > 1){
    fprintf(stderr, "%s\n", message);
  }
#endif
}


void debugvv(char* message){
#ifndef NDEBUG
  if(_debug_verbosity > 2){
    fprintf(stderr, "%s\n", message);
  }
#endif
}

#ifdef __cplusplus
}
#endif