//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim


// main() parses the CLI flags and determines what to do:
// print help, run mumax in master, slave or daemon mode. 

import (
	"flag"
	"fmt"
	"os"
	"path"
	"runtime"
)

// WARNING: most flags added here will need to be passed on to a deamon's child process
// after adding a flag, edit daemon.go accordingly!
var (
	help              *bool   = flag.Bool("help", false, "Print a help message and exit.")
	example           *string = flag.String("example", "", "Create an example input file. E.g.: -example=file.in")
	stdin             *bool   = flag.Bool("stdin", false, "Read input from stdin instead of file. Specify a dummy input file name to determine the output directory name.")
	slave             *bool   = flag.Bool("slave", false, "When as child of another process")
	silent            *bool   = flag.Bool("silent", false, "Do not show simulation output on the screen, only save to output.log")
	daemon            *bool   = flag.Bool("daemon", false, "Watch directories for new input files and run them automatically.")
	random            *bool   = flag.Bool("random", false, "With daemon: run input files in random order.")
	watch             *int    = flag.Int("watch", 60, "With -daemon, re-check for new input files every N seconds. -watch=0 disables watching, program exits when no new input files are left.")
	walltime          *int    = flag.Int("walltime", 0, "With -daemon, keep the deamon alive for N hours. Handy for nightly runs. -walltime=0 (default) runs the daemon forever.")
	verbosity         *int    = flag.Int("verbosity", 2, "Control the debug verbosity (0 - 3)")
	gpuid             *int    = flag.Int("gpu", 0, "Select a GPU when more than one is present. Default GPU = 0") //TODO: also for master
	threads           *int    = flag.Int("threads", 0, "Set the number of threads for the selected device (GPU or CPU). \"0\" means automatically set.")
	updatedb          *int    = flag.Int("updatedisp", 200, "Update the terminal output every x milliseconds")
	wisdir            *string = flag.String("wisdom", defaultWisdomDir(), "Absolute directory to store cached kernels. \"\" disables caching")
	flag_checkversion *bool   = flag.Bool("check-version", true, "Check for now version at startup")
	// 	dryrun    *bool   = flag.Bool("dryrun", false, "Go quickly through the simulation sequence without calculating anything. Useful for debugging") // todo implement
	//  server    *bool   = flag.Bool("server", false, "Run as a slave node in a cluster")
	//  port      *int    = flag.Int("port", 2527, "Which network port to use")
	//  transport *string = flag.String("transport", "tcp", "Which transport to use (tcp / udp)")
)

// called by main.main()
func Main() {
	defer crashreport()                 // if we crash, catch it here and print a nice crash report
	defer fmt.Print(RESET + SHOWCURSOR) // make sure the cursor does not stay hidden if we crash

	flag.Parse()

	if *help {
		Help()
		os.Exit(0)
	}

	if *example != "" {
		Example(*example)
		os.Exit(0)
	}

	Verbosity = *verbosity
	if *daemon {
		DaemonMain()
		return
	}

	if *slave {
		main_slave()
		return
	}

	main_master()
}


func PrintInfo() {
	//	fmt.Println("Running on " + s.Backend.String())
	//	fmt.Println("Max threads: ", s.maxthreads())
	fmt.Println("Go version: ", runtime.Version())
}


// TODO: move to iotool
// Removes a filename extension.
// I.e., the part after the dot, if present.
func RemoveExtension(str string) string {
	ext := path.Ext(str)
	return str[:len(str)-len(ext)]
}

// Removes a filename path.
// I.e., the part before the last /, if present.
// TODO: remove
func RemovePath(str string) string {
	return path.Base(str)
}

// Complementary function of parentDir
// Removes the path in front of the file name.
// I.e., the part before the last /, if present, is removed.
func Filename(str string) string {
	return path.Base(str)
}

// Returns the parent directory of a file.
// I.e., the part after the /, if present, is removed.
// If there is no explicit path, "." is returned.
func ParentDir(str string) string {
	base := path.Base(str)
	return str[:len(str)-len(base)]
}


// when running in "slave" mode, i.e. accepting commands over the network as part of a cluster
// func main_slave() {
// 	server := NewDeviceServer(*device, *transport, *port)
// 	server.Listen()
// }

// This function is deferred from Main(). If a panic()
// occurs, it prints a nice explanation and asks to
// mail the crash report.
// TODO: we need to wrap sim in the error value so
// we can report to its log
func crashreport() {
	error := recover()
	if error != nil {
		switch t := error.(type) {
		default:
			crash(error)
		}
	}
}

func fail() {
	//   err2 := os.Rename(running, sim.outputdir+"/failed")
	//     if err2 != nil {
	//       fmt.Fprintln(os.Stderr, err2)
	//     }
}

func crash(error interface{}) {
	//   err2 := os.Rename(running, sim.outputdir+"/crashed")
	//     if err2 != nil {
	//       fmt.Fprintln(os.Stderr, err2)
	//     }

	fmt.Fprintln(os.Stderr,
`

---------------------------------------------------------------------
Aw snap, the program has crahsed.
If you would like to see this issue fixed, please mail a bugreport to
Arne.Vansteenkiste@UGent.be and/or Ben.VandeWiele@UGent.be.
Be sure to include the output of your terminal, both the parts above
and below this message (in most terminals you can copy the output
with Ctrl+Shift+C).
---------------------------------------------------------------------

`)
	panic(error)
}
