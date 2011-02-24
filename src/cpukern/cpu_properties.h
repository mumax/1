/*
 *  This file is part of MuMax, a high-performance micromagnetic simulator.
 *  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele.
 *  Use of this source code is governed by the GNU General Public License version 3
 *  (as published by the Free Software Foundation) that can be found in the license.txt file.
 *
 *  Note that you are welcome to modify this code under condition that you do not remove any 
 *  copyright notices and prominently state that you modified it, giving a relevant date.
 */

/**
 * @file
 * Accesses the GPU's hardware properties
 *
 * @author Arne Vansteenkiste
 */
#ifndef cpu_properties_h
#define cpu_properties_h

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif


/// Prints the properties of the used CPU
void cpu_print_properties(FILE* out  ///< stream to print to
);


/// Prints to stdout
/// @see print_device_properties()
void cpu_print_properties_stdout();


#ifdef __cplusplus
}
#endif
#endif
