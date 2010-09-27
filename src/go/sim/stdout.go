package sim

// This file implements functions for writing to stdout/stderr
// and simultaneously to a log file. (Unix "tee" functionality)

import (
	"os"
	"io"
	"fmt"
)

func (sim *Sim) Print(msg ...interface{}){
  fmt.Fprint(sim.stdout, msg)
}

func (sim *Sim) Println(msg ...interface{}){
  fmt.Fprintln(sim.stdout, msg)
}

func (sim *Sim) Errorln(msg ...interface{}){
  fmt.Fprintln(sim.stderr, msg)
}

// TODO it would be easier if we had an ansi escape code filter
func (sim *Sim) Warn(msg ...interface{}) {
  fmt.Fprint(os.Stdout, BOLD)
  sim.Print("WARNING: ")
  sim.Print(msg)
  fmt.Fprint(os.Stdout, RESET+ERASE) // Erase rest of line
  sim.Println()
}

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
    outname := outputdir+"/output.log"
		outfile, err := os.Open(outname, os.O_CREATE|os.O_TRUNC, 0666)
		if err != nil {
			// We can not do much more than reporting that we can not save the output,
			// it's not like we can put this message in the log or anything...
			fmt.Fprintln(os.Stderr, err)
		} else {
			if sim.stdout == nil {
				sim.stdout = outfile
			} else {
				sim.stdout = io.MultiWriter(outfile, sim.stdout)
			}
			sim.Println("Opened log file: ",  outname)
		}

    errname := outputdir+"/error.log"
		errfile, err2 := os.Open(errname, os.O_CREATE|os.O_TRUNC, 0666)
		if err != nil {
			fmt.Fprintln(os.Stderr, err2)
		} else {
			if sim.stderr == nil {
				sim.stderr = errfile
			} else {
				sim.stderr = io.MultiWriter(errfile, sim.stderr)
			}
		}
		sim.Println("Opened error log file: ",  errname)
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
