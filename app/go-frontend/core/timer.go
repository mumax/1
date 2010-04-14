package core

/*
#include "../../../core/timer.h"
*/
import "C"

func TimerPrintAll(){
  C.timer_printall();
}

// func TimerPrintDetail(){
//   C.timer_printdetail();
// }
