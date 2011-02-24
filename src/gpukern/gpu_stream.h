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
 * Central place where GPU streams are managed
 * 
 * @author Arne Vansteenkiste
 */
#ifndef gpu_stream_h
#define gpu_stream_h


#ifdef __cplusplus
extern "C" {
#endif

#define MAXGPUSTREAMS 32

/**
 * Gets a stream from the pool of streams.
 */
cudaStream_t gpu_getstream();

///@internal
void gpu_init_stream();

#ifdef __cplusplus
}
#endif
#endif
