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

#define MAXGPUSTREAMS 16

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
