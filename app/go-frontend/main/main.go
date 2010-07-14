package main

import(
  "sim"
  "refsh"
  
)

func main(){
  refsh := refsh.New()
  refsh.Add("verbosity", Verbosity)
  refsh.Add("verbosity", Verbosity)
}


func Verbosity(verbosity int){
  sim.Verbosity = verbosity
}
