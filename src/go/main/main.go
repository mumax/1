// TODO automatic backend selection

// TODO read magnetization + scale
// TODO Time-dependent quantities

// TODO Nice output / table
// TODO draw output immediately
// TODO movie output

package main

import (
	. "sim"
	"refsh"
	"os"
	"fmt"
	"flag"
	"io"
)

var(
  server *bool = flag.Bool("server", false, "Run as a slave node in a cluster")
  )
  
func main() {
  flag.Parse()
  
	if *server{
    main_slave()
  }else{
    main_master()
  }
}

// when running in the normal "master" mode, i.e. given an input file to process locally
func main_master(){
  if flag.NArg() == 0 {
    fmt.Fprintln(os.Stderr, "No input files.")
    os.Exit(-1)
  }

  for i := 0; i < flag.NArg(); i++ {
    in, err := os.Open(flag.Arg(i), os.O_RDONLY, 0666)
    if err != nil {
      fmt.Fprintln(os.Stderr, err)
      os.Exit(-2)
    }
    defer in.Close()
    exec(in)
  }
}

// when running in "slave" mode, i.e. accepting commands over the network as part of a cluster
func main_slave(){
  SimServerMain() // is in package sim for visibility reasons
}


func exec(in io.Reader) {
	sim := NewSim()

	refsh := refsh.New()
	refsh.CrashOnError = true
	refsh.AddAllMethods(sim)
	refsh.Exec(in)
}
