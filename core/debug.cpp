#include "debug.h"

#ifdef __cplusplus
extern "C" {
#endif

int _debug_verbosity = 3;   ///@todo default should be 1

int debug_getverbosity(){
  return _debug_verbosity;
}

#ifdef __cplusplus
}
#endif