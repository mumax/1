/**
 * @file
 *
 *
 * @author Arne Vansteenkiste
 */
#ifndef DEBUG_H
#define DEBUG_H

#include "assert.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Setting the verbosity controls how many debug messages are displayed.
 *
 *  - The default, 1, means only debug() messages are shown.
 *  - Level 2 is more verbose: also debugv() messages are shown.
 *  - Level 3 means full verbosity: even debugvv() messages are shown,
 * which may cause a huge amount of output.
 *  - Level 0 means be silent, show no messages at all.
 *
 * @note Do not worry about performance. When compiling with NDEBUG defined,
 * all debug messages are suppressed and the compiler should throw away
 * the calls to empty functions.
 */
void debug_verbosity(int level);

/**
 *  Prints a debug message that is considered generally useful and displayed by default
 */
void debug(char* message);

/**
 * Prints a debug message that is only useful when we are specifically debugging.
 * It is not displayed by default, but only when the verbosity level is at 2 or higher
 */
void debugv(char* message);

/**
 * Prints a debug message that is annoying or printed a lot, like inside a loop.
 * It is not displayed by default, but only when the verbosity level is at 3 or higher
 */
void debugvv(char* message);

#ifdef __cplusplus
}
#endif
#endif
