package sim

import "log"

func assert(b bool){
  if !b{
    log.Crash("assertion failed");
  }
}
