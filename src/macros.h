#ifndef X
#define X 0
#define Y 1
#define Z 2
#endif





#define NONE 0

// Anisotropy types ***************
/// Possible value for anisType. Means no anisotropy
#define ANIS_NONE     0
/// Possible value for anisType. Means uniaxial anisotropy.
#define ANIS_UNIAXIAL 1
/// Possible value for anisType. Means cubic anisotropy.
#define ANIS_CUBIC    2

// does not belong here: should be able to combine edge anis with crystal anis.
// Possible value for anisType. Means shape anisotropy.
// #define ANIS_EDGE     3
// ********************************


// Solver types *******************
    /// Possible value for solverType. Simple Euler method
  #define SOLVER_EULER 1
    /// Possible value for solverType. 2nd order Heun method
  #define SOLVER_HEUN 2
    /// Possible value for solverType. 4th order Runge-Kutta
  #define SOLVER_RK4 4
    /// Possible value for solverType. 5th order Dormand-Prince with adaptive step size
  #define SOLVER_DP45 45
    /// Possible value for solverType. Ben Van de Wiele's forward semi-analytical solver
  #define SOLVER_ANAL_FW 128
    /// Possible value for solverType. Ben Van de Wiele's predictor/corrector semi-analytical solver
  #define SOLVER_ANAL_PC 256
// ********************************


// Exchange types *****************
  /// 6-neighbour exchange (2D and 3D geometry).
#define EXCH_6NGBR 6
// ********************************


// Kernel types *******************
#define KERNEL_DIPOLE 1
  /// Quantities only dependent on Y and Z coordinate, invariance in X direction. Includes exchange
#define KERNEL_MICROMAG2D 2

  /// Classical 3D micromagnetic kernel. Can also be used for 2D simulations (1 cell in X). Includes exchange.
#define KERNEL_MICROMAG3D 3
// *********************************







#ifndef PI
#define PI 3.14159265f
#endif

/// integer division, but rounded UP
#define divUp(x, y) ( (((x)-1)/(y)) +1 )

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
#define _debug_verbosity 2

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