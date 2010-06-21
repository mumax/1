/**
 * @file
 * Using sub-processes is a universal way to glue together programs written in different languages.
 * The main program can communticate with its sub-processes by sending data over UNIX pipes. 
 * Using sub-processes is mainly useful in the non performance-critical stages of a program,
 * as there is some overhead associated to the creation of each new process.
 * 
 * This file provides some utilities for working with pipes. 
 * E.g., pipe_tensor() executes a command and returns its standard output directly as a tensor,
 * thus releaving the user to manully open the pipe, read the tensor, check for errors, etc.
 *
 * @todo pipe_kernel(msat, aexch, size, ...) to get a micromagnetic kernel
 * @todo pipe_config() for an initial configuration
 * @todo pipe_output() for post-processing
 *
 * @author Arne Vansteenkiste
 */
#ifndef PIPES_H
#define PIPES_H

#include "tensor.h"
#include "param.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Executes a command that should output a tensor to stdout, and returns the tensor.
 * The tensor should be passed in the standard tensor format (see tensor.h).
 */
tensor* pipe_tensor(char* command);

tensor* pipe_kernel(param* params);

#ifdef __cplusplus
}
#endif
#endif