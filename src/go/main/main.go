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
	"runtime"
)

var (
	server    *bool   = flag.Bool("server", false, "Run as a slave node in a cluster")
	verbosity *int    = flag.Int("verbosity", 1, "Control the debug verbosity (0 - 3)")
	port      *int    = flag.Int("port", 2527, "Which network port to use")
	transport *string = flag.String("transport", "tcp", "Which transport to use (tcp / udp)")
	device    *string = flag.String("device", "gpu", "The default computing device to use with -server") //TODO: also for master
  updatedb  *int    = flag.Int("updatedisp", 100, "Update the terminal output every x milliseconds")
)

func main() {
	flag.Parse()

	Verbosity = *verbosity

	if *server {
		main_slave()
	} else {
		main_master()
	}
}

// when running in the normal "master" mode, i.e. given an input file to process locally
func main_master() {

  Debugv("Locked OS thread")
  runtime.LockOSThread()
  
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "No input files.")
		os.Exit(-1)
	}

  UpdateDashboardEvery = int64(*updatedb * 1000 * 1000)
  
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
func main_slave() {
	server := NewDeviceServer(*device, *transport, *port)
	server.Listen()
}


func exec(in io.Reader) {
	sim := NewSim()

	refsh := refsh.New()
	refsh.CrashOnError = true
	refsh.AddAllMethods(sim)
	refsh.Exec(in)
	sim.TimerPrintDetail()
	sim.PrintTimer(os.Stdout)
}
