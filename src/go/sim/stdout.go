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

// Prints to stderr (unless sim.silent=true) and also to error.log
func (sim *Sim) Errorln(msg ...interface{}) {
	if !sim.silent {
		fmt.Fprintln(os.Stderr, msg)
	}
	fmt.Fprintln(sim.err, msg)
}

// Prints to stdout (unless sim.silent=true) in bold font
// and also to output.log in plain text
func (sim *Sim) Warn(msg ...interface{}) {
	sim.Escape(BOLD)
	sim.Print("WARNING: ")
	sim.Print(msg)
	sim.Escape(RESET + ERASE + "\n") // Erase rest of line
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

// Initiates the stderr and stdout files of sim.
// If not silent, they will print on the screen.
// If log=true, they will also log to outputdir/output.log
// and outputdir/error.log in a Unix "tee"-like fashion.
// func (sim *Sim) initWriters(outputdir string, silent, log bool) {
//
//  if !silent {
//    sim.stdout = os.Stdout
//    sim.stderr = os.Stderr
//  }
//
//  if log {
//     outname := outputdir+"/output.log"
//    outfile, err := os.Open(outname, os.O_CREATE|os.O_TRUNC, 0666)
//    if err != nil {
//      // We can not do much more than reporting that we can not save the output,
//      // it's not like we can put this message in the log or anything...
//      fmt.Fprintln(os.Stderr, err)
//    } else {
//      if sim.stdout == nil {
//        sim.stdout = outfile
//      } else {
//        sim.stdout = io.MultiWriter(outfile, sim.stdout)
//      }
//      sim.Println("Opened log file: ",  outname)
//    }
//
//     errname := outputdir+"/error.log"
//    errfile, err2 := os.Open(errname, os.O_CREATE|os.O_TRUNC, 0666)
//    if err != nil {
//      fmt.Fprintln(os.Stderr, err2)
//    } else {
//      if sim.stderr == nil {
//        sim.stderr = errfile
//      } else {
//        sim.stderr = io.MultiWriter(errfile, sim.stderr)
//      }
//    }
//    sim.Println("Opened error log file: ",  errname)
//  }
//
//   if sim.stdout == nil{ sim.stdout = new(DevNull)}
//   if sim.stderr == nil{ sim.stderr = new(DevNull)}
//
//
// }
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

	errname := sim.outputdir + "/error.log"
	errfile, err2 := os.Open(errname, os.O_WRONLY|os.O_CREAT, 0666)
	if err != nil {
		fmt.Fprintln(os.Stderr, err2)
		sim.err, _ = os.Open(os.DevNull, 0, 0666)
	} else {
		sim.err = errfile
	}
}
// 
// // a bottomless sink for bytes
// type DevNull struct{
// }
// 
// func (null *DevNull) Write(p []byte) (n int, err os.Error){
//   n = len(p)
//   err = nil
//   return
// }
