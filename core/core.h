/**
 * @mainpage
 * 
 * Micromagnetic/Spin-Lattice Simuations on the GPU 
 *
 * The core library (trunk/core) contains the building blocks for running magnetic simulations on the GPU.
 * The high-level building blocks provide:
 * 	- convolutions	(gpuconv1.h, ...)
 *	- time stepping	(gpueuler.h, gpuheun.h, ...)
 *	- micromagnetic kernels
 * 	- unit conversion (units.h)
 *	- tensor utilities (tensor.h)
 *	- performance measurment (timer.h)
 *
 * These building blocks are used by the main simulation programs in trunk/app
 *
 * Some lower-level functions that are used by the above building blocks include:
 *	- FFT's (gputil.h)
 *	- GPU data manipulation/communication (gputil.h)
 *	- communication with other processes (pipes.h)
 *	- ...
 *
 * These are usually not directly needed by a main simulation program.
 *
 * Aditionally, some auxilary programs are present in trunk/app.
 * Their binaries are put in trunk/bin, which should be added to your $PATH.
 *
 * These programs include:
 *	- tensor, for manipulating or post-processing data in the tensor format (see tensor.h)
 *	- kernel, for generating micromagnetic kernels
 *	- config, for generating initial magnetic configurations
 *	- ...
 *
 * These command-line programs can be handy for the user, 
 * or can be directly called from a main simulation program (e.g., using pipes.h).
 *
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */


/**
 * @file
 * Includes all .h files in core/
 *
 * @author Arne Vansteenkiste
 */

#include gpuconv1.h
#include gpueuler.h
#include gpuheun.h
#include gpurk4.h
#include gputil.h
#include tensor.h
#include pipes.h
#include timer.h
