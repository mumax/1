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
 *
 * @author Arne Vansteenkiste
 * @author Ben Van de Wiele
 */
#ifndef gpu_reduction_h
#define gpu_reduction_h

#ifdef __cplusplus
extern "C" {
#endif

#define REDUCE_ADD 1
#define REDUCE_MAX 2
#define REDUCE_MAXABS 3
#define REDUCE_MIN 4
#define REDUCE_SUMABS 5

/// Reduces the input (array on device)
float gpu_reduce(int operation,     ///< REDUCE_ADD, REDUCE_MAX, ... 
                 float* input,      ///< input data on device
                 float* devbuffer,  ///< device buffer of size "blocks"
                 float* hostbuffer, ///< host buffer of size "blocks"
                 int blocks,        ///< blocks * threadsPerBlock * 2 = N
                 int threadsPerBlock,///< threads per thread block (maximum is device dependent)
                 int N              ///< input size
                 );

float gpu_sum(float* data, int size);
  
/// @internal Low-level partial sum function
void gpu_partial_sums(float* input,        ///< input data. size = N @todo check if it gets overwritten
                    float* output,       ///< partially reduced data, usually reduced further on CPU. size = blocks
                    int blocks,          ///< patially reduce in X blocks, partial results in output. blocks = divUp(N, threadsPerBlock*2)
                    int threadsPerBlock, ///< use X threads per block: @warning must be < N
                    int N                ///< size of input data, must be > threadsPerBlock
                    );

#ifdef __cplusplus
}
#endif
#endif
