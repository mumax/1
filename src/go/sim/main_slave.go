//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements the main() for mumax running in slave mode.
// Usually, the slave takes input from stdin that is piped from a
// master process.
// The master process can be mumax running in master mode or, e.g.,
// python or java wrappers.

import (
	"os"
	"flag"
	"fmt"
	"time"
	"refsh"
)

// Do the actual work, do not start sub-processes
func main_slave() {

	if flag.NArg() == 0 {
		NoInputFiles()
		os.Exit(-1)
	}

	UpdateDashboardEvery = int64(*updatedb * 1000 * 1000)

	// Process all input files
	for i := 0; i < flag.NArg(); i++ {
		infile := flag.Arg(i)
		var in *os.File
		var err os.Error
		if *stdin {
			in = os.Stdin
		} else {
			in, err = os.Open(infile, os.O_RDONLY, 0666)
			if err != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(-2)
			}
		}
		defer in.Close()

		//TODO it would be safer to abort when the output dir is not empty
		outfile := RemoveExtension(infile) + ".out"

		// Set the device
		var backend *Backend
		if *cpu {
			backend = CPU

		} else {
			backend = GPU
			backend.SetDevice(*gpuid)
		}

		sim := NewSim(outfile, backend)
		defer sim.Close()

		sim.wisdomdir = *wisdir

		// file "running" indicates the simulation is running
		running := sim.outputdir + "/running"
		runningfile, err := os.Open(running, os.O_WRONLY|os.O_CREATE, 0666)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
		}
		fmt.Fprintln(runningfile, "This simulation was started on:\n", time.LocalTime(), "\nThis file is renamed to \"finished\" when the simulation is ready.")
		runningfile.Close()

		sim.silent = *silent
		refsh := refsh.New()
		refsh.CrashOnError = true
		refsh.AddAllMethods(sim)
		refsh.Output = sim
		refsh.Exec(in)

		// We're done
		err2 := os.Rename(running, sim.outputdir+"/finished")
		if err2 != nil {
			fmt.Fprintln(os.Stderr, err2)
		}

		// Idiot-proof error reports
		if refsh.CallCount == 0 {
			sim.Errorln("Input contained no commands.")
		}
		if !sim.BeenValid {
			sim.Errorln("Input contained no commands to make the simulation run (e.g., \"run\").")
		}
		// The next two lines cause a nil pointer panic when the simulation is not fully initialized
		//if sim.BeenValid && Verbosity > 1 {
		//	down()
		//	down()
		//	down()
		//	down()
		//	down()

			//sim.TimerPrintDetail()
			//sim.PrintTimer(os.Stdout)
		//}

		// TODO need to free sim

	}
}
