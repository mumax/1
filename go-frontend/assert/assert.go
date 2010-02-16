package assert

import(
  "log";
);


func Assert(test bool){
  if !test{
    log.Crash("Assertion failed");
  }
}


func AssertMsg(test bool, msg string){
  if !test{
    log.Crash(msg);
  }
}
