package sim

// This file implements functions for writing to stdout/stderr
// and simultaneously to a log file. (Unix "tee" functionality)

func (sim *Sim) initWriters(outputdir string, silent, log bool){
  outwriters := [2]io.Writer{}
  errwriters := [2]io.Writer{}
  i := 0

  if !silent{
    outwriters[i] = os.Stdout
    errwriters[i] = os.Stderr
    i++
  }

  if log{
    outwriters[i] = 
  }
}

