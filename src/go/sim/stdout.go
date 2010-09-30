//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements functions for writing to stdout/stderr
// and simultaneously to a log file.
// Inside simulation code, use *Sim.Println() etc, not fmt.Println().

import (
	"os"
	"fmt"
)

// Prints to stdout (unless sim.silent=true) and also to output.log
func (sim *Sim) Print(msg ...interface{}) {
	if !sim.silent {
		fmt.Fprint(os.Stdout, msg)
	}
	fmt.Fprint(sim.out, msg)
}

// Prints to stdout (unless sim.silent=true) and also to output.log
func (sim *Sim) Println(msg ...interface{}) {
	sim.Print(msg)
	sim.Print("\n")
}

// Prints to stderr (unless sim.silent=true) and also to output.log
func (sim *Sim) Errorln(msg ...interface{}) {
	if !sim.silent {
		sim.Escape(BOLD + RED)
		fmt.Fprintln(os.Stderr, msg)
		sim.Escape(RESET)
	}
	fmt.Fprintln(sim.out, msg)
}

// Prints to stdout (unless sim.silent=true) in bold font
// and also to output.log in plain text
func (sim *Sim) Warn(msg ...interface{}) {
	sim.Escape(BOLD + RED)
	sim.Print("WARNING: ")
	sim.Print(msg)
	sim.Escape(RESET + ERASE) // Erase rest of line
	sim.Println()
}

// Prints to stdout (unless sim.silent=true) but not to output.log.
// Use this for printing ANSI escape characters that should not
// appear in the output file.
func (sim *Sim) Escape(msg ...interface{}) {
	if !sim.silent {
		fmt.Fprint(os.Stdout, msg)
	}
}


func (sim *Sim) initWriters() {

	outname := sim.outputdir + "/output.log"
	outfile, err := os.Open(outname, os.O_WRONLY|os.O_CREAT, 0666)
	if err != nil {
		// We can not do much more than reporting that we can not save the output,
		// it's not like we can put this message in the log or anything...
		fmt.Fprintln(os.Stderr, err)
		// sim.out should not be nil so we don't crash on output,
		// so we dump to /dev/null (we are very unlikely to reach this point)
		sim.out, _ = os.Open(os.DevNull, 0, 0666)
	} else {
		sim.out = outfile
	}

	// 	errname := sim.outputdir + "/error.log"
	// 	errfile, err2 := os.Open(errname, os.O_WRONLY|os.O_CREAT, 0666)
	// 	if err != nil {
	// 		fmt.Fprintln(os.Stderr, err2)
	// 		sim.err, _ = os.Open(os.DevNull, 0, 0666)
	// 	} else {
	// 		sim.err = errfile
	// 	}
}
