#ifndef LIBTENSOR_H
#define LIBTENSOR_H

#include <stdio.h>

// allow inclusion in C++ code
#ifdef __cplusplus
extern "C" {
#endif


/* Define SINGLE_PRECISSION or DOUBLE_PRECISSION. Can be overridden via a command-line flag. */
#ifndef SINGLE_PRECISSION
#ifndef DOUBLE_PRECISSION

#define SINGLE_PRECISSION
//#define DOUBLE_PRECISSION

#endif
#endif


/**
 * We will probably do everything with single-precission floating point data. Nevertheless we use 'real' instead of float so that later we might still re-define it as double. 
 */
#ifdef SINGLE_PRECISSION
typedef float real;
#endif

#ifdef DOUBLE_PRECISSION
typedef double real;
#endif



#ifdef __cplusplus
}
#endif

#endif