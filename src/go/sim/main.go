package sim

// TODO: ALL OF THIS CODE SHOULD BE MOVED INTO THE sim PACKAGE

// TODO automatic backend selection

// TODO read magnetization + scale
// TODO Time-dependent quantities

// TODO Nice output / table
// TODO draw output immediately
// TODO movie output

import (
	"flag"
	"fmt"
	"os"
	"refsh"
	"runtime"
)

var (
	daemon    *bool   = flag.Bool("daemon", false, "Run in the background and watch a directory for input files to process.")
	server    *bool   = flag.Bool("server", false, "Run as a slave node in a cluster")
	verbosity *int    = flag.Int("verbosity", 2, "Control the debug verbosity (0 - 3)")
	port      *int    = flag.Int("port", 2527, "Which network port to use")
	transport *string = flag.String("transport", "tcp", "Which transport to use (tcp / udp)")
	device    *string = flag.String("device", "gpu", "The default computing device to use with -server") //TODO: also for master
	updatedb  *int    = flag.Int("updatedisp", 100, "Update the terminal output every x milliseconds")
	dryrun    *bool   = flag.Bool("dryrun", false, "Go quickly through the simulation sequence without calculating anything. Useful for debugging") // todo implement
)

// to be called by main.main()
func Main() {
	defer crashreport()                 // if we crash, catch it here and print a nice crash report
	defer fmt.Print(RESET + SHOWCURSOR) // make sure the cursor does not stay hidden if we crash

	flag.Parse()
	Verbosity = *verbosity
	if *daemon {
		DaemonMain()
		return
	}

	// 	if *server {
	// 		main_slave()
	// 	} else {
	main_master()
	// 	}
}

// when running in the normal "master" mode, i.e. given an input file to process locally
func main_master() {

	Debugvv("Locked OS thread")
	runtime.LockOSThread()

	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "No input files.")
		os.Exit(-1)
	}

	UpdateDashboardEvery = int64(*updatedb * 1000 * 1000)

	// Process all input files
	for i := 0; i < flag.NArg(); i++ {
		infile := flag.Arg(i)
		in, err := os.Open(infile, os.O_RDONLY, 0666)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(-2)
		}
		defer in.Close()

		//TODO it would be safer to abort when the output dir is not empty
		sim := NewSim(removeExtension(infile) + ".out")
		defer sim.out.Close()
		refsh := refsh.New()
		refsh.CrashOnError = true
		refsh.AddAllMethods(sim)
		refsh.Output = sim
		refsh.Exec(in)

		// Idiot-proof error reports
		if refsh.CallCount == 0 {
			sim.Errorln("Input file contains no commands.")
		}
		if !sim.BeenValid {
			sim.Errorln("Input file does not contain any commands to make the simulation run. Use, e.g., \"run\".")
		}
		// The next two lines cause a nil pointer panic when the simulation is not fully initialized
		if sim.BeenValid {
			sim.TimerPrintDetail()
			sim.PrintTimer(os.Stdout)
		}

		// TODO need to free sim

	}
}

// Removes a filename extension.
// I.e., the part after the dot, if present.
func removeExtension(str string) string {
	dotpos := len(str) - 1
	for dotpos >= 0 && str[dotpos] != '.' {
		dotpos--
	}
	return str[0:dotpos]
}

// when running in "slave" mode, i.e. accepting commands over the network as part of a cluster
// func main_slave() {
// 	server := NewDeviceServer(*device, *transport, *port)
// 	server.Listen()
// }

// This function is deferred from Main(). If a panic()
// occurs, it prints a nice explanation and asks to
// mail the crash report.
func crashreport() {
	error := recover()
	if error != nil {
		panic(error)
	}
}
