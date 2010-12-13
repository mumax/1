package mypack

/*
#include <stdio.h>
void c_hello(){
  printf("Hello from C\n");
}
*/
import "C"

func hello(){
  C.c_hello()
}
