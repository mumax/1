/**
 * @file 
 *
 * A timer for benchmarking and profiling.
 *
 * A timer is started/stopped with a tag to identify what is being timed, e.g.: timer_start("fft"); do_fft(); timer_stop("fft");
 * The time can then be obtained with timer_get("fft"), or printed to the screen with timer_print("fft").
 *
 * A timer can be started and stopped multiple times, which will accumulate the total time.
 *
 * Multiple timers with different tags can safely be running at the same time. 
 * Everyting can be printed to the screen with timer_printall();
 * 
 * @note The CPU time is measured, not the walltime. This means that the time spent waiting for other processes or I/O does not count.
 *
 * @author Arne Vansteenkiste
 */
#ifndef TIMER_H
#define TIMER_H

// allow inclusion in C++ code
#ifdef __cplusplus
extern "C" {
#endif

#include <stdlib.h>
#include <stdio.h>

/** Starts the timer for the tag */
void timer_start(const char* tag	///< identifies what is being timed
		 );
/** Stops the timer for the tag */
void timer_stop (const char* tag	///< identifies what is being timed
		);
/** Returns the time in seconds. timer_stop() should be called first. */
double timer_get(const char* tag	///< identifies what is being timed
		);
/** Prints the time to stderr with format: "tag: xxx s" */
void timer_print(const char* tag	///< identifies what is being timed
		);
/** Prints all the timers that are registered */
void timer_printall();

#ifdef __cplusplus
}
#endif

#endif