package main

import(
  "sim"
  "refsh"
  
)

func main(){
  printInfo()
  startRefsh()
}

func printInfo(){
  fmt.Println(
    "    <program>  Copyright (C) <year>  <name of author>"
    "    This is free software, and you are welcome to redistribute it"
    "    under certain conditions; see licence.txt for details."
  )
}

func startRefsh(){
  refsh := refsh.New()
  refsh.Add("verbosity", Verbosity)  
}

func Verbosity(verbosity int){
  sim.Verbosity = verbosity
}
