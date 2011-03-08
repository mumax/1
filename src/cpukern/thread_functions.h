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
 * This file provides the thread functions used on cpu
 *
 * @author Ben Van de Wiele
 */

#ifndef THREAD_FUNCTIONS_H
#define THREAD_FUNCTIONS_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct{

  int N_threads;
  
  pthread_mutex_t *threads_mutex;
  pthread_mutex_t done_mutex;
  pthread_mutex_t main_mutex;
  pthread_cond_t *threads_cond;
  pthread_cond_t done_cond;
  pthread_t *threads;
  int threads_done;

}thread_data;



// Global variables

  extern void *func_arg;
  extern void (*func_ptr)(int);
  extern thread_data *T_data;
  

  

void init_Threads(int Nthreads          ///>number of threads used in the simulation
                  );

void thread_Wrapper(void (*)(int)       ///>void pointer pointing to a function with one int argument (the id)
                    );

void thread_done();

void *thread_function(void *);          ///> the function which every thread is executing

void init_start_stop(int *start,
                     int *stop,
                     int id,
                     int N
                     );
#ifdef __cplusplus
}
#endif
#endif
