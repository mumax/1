package sim

// This file implements functions for writing to stdout/stderr
// and simultaneously to a log file. (Unix "tee" functionality)

import (
	"os"
	"io"
	"fmt"
)

// Initiates the stderr and stdout files of sim.
// If not silent, they will print on the screen.
// If log=true, they will also log to outputdir/output.log
// and outputdir/error.log in a Unix "tee"-like fashion.
func (sim *Sim) initWriters(outputdir string, silent, log bool) {

	if !silent {
		sim.stdout = os.Stdout
		sim.stderr = os.Stderr
	}

	if log {
		outfile, err := os.Open(outputdir+"/output.log", os.O_CREATE|os.O_TRUNC, 0666)
		if err != nil {
			// We can not do much more than reporting that we can not save the output,
			// it's not like we can put this message in the log or anything...
			fmt.Fprintln(os.Stderr, err)
		} else {
			if sim.stdout == nil {
				sim.stdout = outfile
			} else {
				sim.stdout = io.MultiWriter(sim.stdout, outfile)
			}
		}

		errfile, err2 := os.Open(outputdir+"/error.log", os.O_CREATE|os.O_TRUNC, 0666)
		if err != nil {
			fmt.Fprintln(os.Stderr, err2)
		} else {
			if sim.stderr == nil {
				sim.stderr = errfile
			} else {
				sim.stderr = io.MultiWriter(sim.stderr, errfile)
			}
		}
	}

	  if sim.stdout == nil{ sim.stdout = new(DevNull)}
	 if sim.stderr == nil{ sim.stderr = new(DevNull)}


}

// a bottomless sink for bytes
type DevNull struct{
}

func (null *DevNull) Write(p []byte) (n int, err os.Error){
  n = len(p)
  err = nil
  return
}
