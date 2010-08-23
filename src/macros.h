#ifndef X
#define X 0
#define Y 1
#define Z 2
#endif


#ifndef PI
#define PI 3.14159265f
#endif


#ifndef debug

/**
 * Setting the verbosity controls how many debug messages are displayed.
 *
 *  - The default, 1, means only debug() messages are shown.
 *  - Level 2 is more verbose: also debugv() messages are shown.
 *  - Level 3 means full verbosity: even debugvv() messages are shown,
 * which may cause a huge amount of output.
 *  - Level 0 means be silent, show no messages at all.
 *
 * Example:
 * @code
    debugv( fprintf(stderr, "size = %d \n", size) );
    beenhere();     // Prints: "Been here: file XXX.c, line YYY (only for verbosity 3)
 * @endcode
 *
 * @note Do not worry about performance. When compiling with verbosity = 0,
 * all debug messages are suppressed.
 */
#define _debug_verbosity 3

#include "assert.h"
#include <stdio.h>




/**
 *  Prints a debug message that is considered generally useful and displayed by default
 */
#define     debug(cmd)       { if(_debug_verbosity > 0){ cmd; } }

/**
 * Prints a debug message that is only useful when we are specifically debugging.
 * It is not displayed by default, but only when the verbosity level is at 2 or higher
 */
#define     debugv(cmd)       { if(_debug_verbosity > 1){ cmd; } }

/**
 * Prints a debug message that is annoying or printed a lot, like inside a loop.
 * It is not displayed by default, but only when the verbosity level is at 3 or higher
 */
#define     debugvv(cmd)       { if(_debug_verbosity > 2){ cmd; } }

/**
 * Prints "Been here: " with the corresponding file and line number.
 * Only for verbosity 3.
 */
#define    beenhere()         { if(_debug_verbosity > 2){ fprintf(stderr, "Been here: %s line %d\n", __FILE__, __LINE__); } }

#endif