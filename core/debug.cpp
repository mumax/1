#include "debug.h"

#ifdef __cplusplus
extern "C" {
#endif

int _debug_verbosity = 1;   /// default is 1

int debug_getverbosity(){
  return _debug_verbosity;
}

#ifdef __cplusplus
}
#endif