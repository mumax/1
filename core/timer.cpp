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

void timer_start(const char* tag){
  timer_table[tag] = clock();
}

void timer_stop(const char* tag){
  timer_total[tag] += (double)(clock() - timer_table[tag]) / (double)CLOCKS_PER_SEC;
}

double timer_get(const char* tag){
  return timer_total[tag];
}

void timer_print(const char* tag){
  cerr << tag << ":" << timer_get(tag) << " s" << endl;
}

void timer_printall(){
  for(map<string, double>::iterator it = timer_total.begin(); it != timer_total.end(); it++){
    timer_print(it->first.c_str());
  }
}

#ifdef __cplusplus
}
#endif