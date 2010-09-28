/**
 * @file 
 *
 * A timer for benchmarking and profiling.
 *
 * A timer is started/stopped with a tag to identify what is being timed, e.g.: 
 * @code
 * timer_start("fft"); 
 * do_fft(); 
 * timer_stop("fft");
 * @endcode
 *
 * At the end of program, call
 * @code
 * timer_printdetail();
 * @endcode
 * To get a detailed overview of how much time is spent in each of the timed sections.
 *
 * The time of an individual section can then be obtained with, e.g.: @code timer_get("fft") @endcode 
 * or printed to the screen with @code timer_print("fft") @endcode.
 *
 * A timer can be started and stopped multiple times, which will accumulate the total time.
 *
 * @note The CPU time is measured, not the walltime. This means that the time spent waiting for other processes or I/O does not count.
 *
 * @author Arne Vansteenkiste
 *
 * @note
 * "We should forget about small efficiencies, say about 97% of the time: 
 * premature optimization is the root of all evil"
 * -- Donald Knuth 
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
void timer_printall(void);

/**
 * Like timer_printall(), but also prints percentages of the total time.
 */
void timer_printdetail(void);

/** 
 * The time elapsed between the first timer_start() and the last timer_stop() call.
 * If all went well, this should be approximately equal to timer_accumulatedtime().
 */
double timer_elapsedtime(void);

/** 
 * The sum of registered times by all tags.
 * If all went well, this should be approximately equal to timer_totaltime().
 * If it is significantly smaller, important portions of the code have probably not been timed,
 * if it is significantly larger, portions may have been double-timed.
 */
double timer_accumulatedtime(void);

#ifdef __cplusplus
}
#endif

#endif