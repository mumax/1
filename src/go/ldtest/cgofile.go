package mypack

/*
#include "mylib.h"
*/
import "C"

func hello() {
	C.c_hello()
}
