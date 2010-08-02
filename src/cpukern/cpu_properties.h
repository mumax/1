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
