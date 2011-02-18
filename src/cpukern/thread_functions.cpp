#include "thread_functions.h"

#ifdef __cplusplus
extern "C" {
#endif

  void *func_arg;
  void (*func_ptr)(int);
  thread_data *T_data;

void init_Threads(int Nthreads){
   
	int id[Nthreads];

  T_data = (thread_data *) calloc(1, sizeof (thread_data));
  T_data->N_threads = Nthreads;
	T_data->threads_done = Nthreads;
	T_data->threads = (pthread_t *) calloc(Nthreads, sizeof(pthread_t));
	T_data->threads_mutex = (pthread_mutex_t *) calloc(Nthreads, sizeof(pthread_mutex_t));
	T_data->threads_cond = (pthread_cond_t *) calloc(Nthreads, sizeof(pthread_cond_t));

	for(int cnt=0; cnt<Nthreads; cnt++){
		pthread_mutex_init(&T_data->threads_mutex[cnt], NULL);
		pthread_cond_init(&T_data->threads_cond[cnt], NULL);
    id[cnt] = cnt;
	}
	pthread_mutex_init(&T_data->done_mutex, NULL);
  pthread_mutex_init(&T_data->done_mutex, NULL);
  pthread_mutex_init(&T_data->main_mutex, NULL);
	pthread_cond_init(&T_data->done_cond, NULL);


	for(int cnt=0; cnt<Nthreads; cnt++){
//    pthread_create(&T_data->threads[cnt], NULL, (void *) thread_function, (void *) &id[cnt]);
    pthread_create(&T_data->threads[cnt], NULL, thread_function, (void *) &id[cnt]);
  }

	pthread_cond_wait(&T_data->done_cond, &T_data->done_mutex);

	pthread_mutex_unlock(&T_data->done_mutex);
  pthread_mutex_unlock(&T_data->main_mutex);

	return;
}


void thread_Wrapper( void (*function)(int)){

  pthread_mutex_lock (&T_data->main_mutex);
  for(int cnt=0; cnt<T_data->N_threads; cnt++)
    pthread_mutex_lock (&T_data->threads_mutex[cnt]);

  func_ptr = function;

  for(int cnt=0; cnt<T_data->N_threads; cnt++){
    pthread_mutex_unlock (&T_data->threads_mutex[cnt]);
    pthread_cond_signal(&T_data->threads_cond[cnt]);
  }

  pthread_cond_wait(&T_data->done_cond, &T_data->main_mutex);
  pthread_mutex_unlock(&T_data->main_mutex);

  return;
}

void *thread_function(void *input){

  int id = *((int *) input);

  pthread_mutex_lock (&T_data->threads_mutex[id]);
	thread_done();

	while(1){
		pthread_cond_wait(&T_data->threads_cond[id], &T_data->threads_mutex[id]);
		(*func_ptr)(id);
		thread_done();
	}

	return (NULL);
}

void thread_done(){

  int Nthreads = T_data->N_threads;

  pthread_mutex_lock (&T_data->done_mutex);
  
  T_data->threads_done--;
  if (!T_data->threads_done){
    T_data->threads_done = Nthreads;
    pthread_mutex_lock(&T_data->main_mutex);
    pthread_mutex_unlock(&T_data->main_mutex);
    pthread_cond_signal(&T_data->done_cond);
  }
  pthread_mutex_unlock (&T_data->done_mutex);

  return;
}

void init_start_stop(int *start, int *stop, int id, int N){

  if (N<T_data->N_threads)
    if (id<N){
      *start = id;
      *stop = id+1;
    }
    else{
      *start = 0;
      *stop = -1;
    }      
  else{  
    *start = id * N/T_data->N_threads;
    if (id!=T_data->N_threads-1)
      *stop = (id+1) * N/T_data->N_threads;
    else
      *stop = N;
  }

  return;
}
#ifdef __cplusplus
}
#endif
