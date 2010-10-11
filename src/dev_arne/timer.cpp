#include "timer.h"
#include <time.h>
#include <string>
#include <map>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

using namespace std;

map<string, long long int> timer_table;
map<string, double> timer_total;

long long int timer_first_start = -1;
long long int timer_last_stop = -1;
time_t timer_start_wall = -1;
time_t timer_stop_wall = -1;

void timer_start(const char* tag){
  if(timer_first_start == -1){
    timer_first_start = clock();
    time(&timer_start_wall);
  }
  timer_table[tag] = clock();
}

void timer_stop(const char* tag){
  timer_total[tag] += (double)(clock() - timer_table[tag]) / (double)CLOCKS_PER_SEC;
  timer_last_stop = clock();
  time(&timer_stop_wall);
}

double timer_get(const char* tag){
  return timer_total[tag];
}

void timer_print(const char* tag){
  fprintf(stderr, "%25s: %6.2lf s", tag, timer_get(tag));
}

void timer_printall(){
  for(map<string, double>::iterator it = timer_total.begin(); it != timer_total.end(); it++){
    timer_print(it->first.c_str());
    cerr << endl;
  }
}

void timer_printdetail(){
  double acc = timer_accumulatedtime();
  double elapsed = timer_elapsedtime();
  double real = difftime(timer_stop_wall, timer_start_wall);
  
  cerr << "elapsed real time: " <<  real << " s" << endl;
  cerr << "elapsed CPU time : " << elapsed << " s" << endl;
  cerr << "measured CPU time: " << acc << " s" << endl;
  
  for(map<string, double>::iterator it = timer_total.begin(); it != timer_total.end(); it++){
    timer_print(it->first.c_str());
    fprintf(stderr, " (%5.2lf %%)\n", 100.0 * timer_get(it->first.c_str()) / elapsed);  
  }
  
  fprintf(stderr, "%25s: %6.2lf s", "[not measured]", elapsed - acc);
  fprintf(stderr, " (%5.2lf %%)\n", 100.0 * (elapsed - acc) / elapsed);
  
  if(acc > elapsed){
    fprintf(stderr, "Warning: the total measured CPU time is larger than the elapsed CPU time, this means portions of the code have been included in more than one timing loop.\n");
  }
}

double timer_elapsedtime(){
  return (double)(timer_last_stop - timer_first_start) / (double)CLOCKS_PER_SEC;
}

double timer_accumulatedtime(){
  double acc = 0.0;
  for(map<string, double>::iterator it = timer_total.begin(); it != timer_total.end(); it++){
    acc += it->second;
  }
  return acc;
}

#ifdef __cplusplus
}
#endif