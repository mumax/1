package sim

import (
	"fmt"
	"os"
	"time"
)

var (
	lastDashUpdate       int64 = 0
	UpdateDashboardEvery int64 = 25 * 1000 * 1000 // in ns
	dashboardNeedsUp     bool  = false
)

func updateDashboard(sim *Sim) {
	/*/*
	  if dashboardNeedsUp{
	    //fmt.Printf(ESC + "2F") // move up N lines
	    up()
	  }
	*/
	fmt.Print(HIDECURSOR)

	T := sim.UnitTime()

	nanotime := time.Nanoseconds()
	if (nanotime - lastDashUpdate) < UpdateDashboardEvery {
		return // too soon to update display yet
	}
	lastDashUpdate = nanotime

	//fmt.Print(HIDECURSOR)
	// Walltime
	time := time.Seconds() - sim.starttime
	fmt.Printf(
		BOLD+"running:"+RESET+"%3dd:%02dh:%02dm:%02ds",
		time/DAY, (time/HOUR)%24, (time/MIN)%60, time%60)
	erase()
	fmt.Println()

	// Time stepping
	fmt.Printf(
		BOLD+"step: "+RESET+"%-11d "+
			BOLD+"time: "+RESET+"%.4es      "+
			BOLD+"Δt:   "+RESET+" %.3es",
		sim.steps, float(sim.time)*T, sim.dt*T)
	erase()
	fmt.Println()

	fmt.Print(BOLD+"IO: "+RESET, sim.autosaveIdx)
	erase()
	fmt.Println()

	fmt.Print(BOLD+"GPU mem: "+RESET, sim.UsedMem()/MiB, " MiB")
	eraseln()

	// Conditions
	//B := sim.UnitField()
	fmt.Printf(
		BOLD+"B:    "+RESET+"(%.3e, %.3e, %.3e)T",
		sim.hext[0], sim.hext[1], sim.hext[2])
	erase()
	fmt.Println()

	up()
	up()
	up()
	up()
	up()
}

func (s *Sim) printMem(){
  // SEGFAULTS !
  //fmt.Println("s", s)
//   fmt.Println("GPU memory used: ", s.UsedMem()/MiB, " MiB")
}

func erase() {
	fmt.Fprint(os.Stdout, ERASE)
}

func eraseln() {
	fmt.Fprintln(os.Stdout, ERASE)
}

func up() {
	fmt.Printf(LINEUP)
}

// ANSI escape sequences
const (
	ESC = "\033["
	// Erase rest of line
	ERASE = "\033[K"
	// Restore cursor position
	RESET = "\033[0m"
	// Bold
	BOLD = "\033[1m"
	// Line up
	LINEUP = "\033[1A"
	// Hide cursor
	HIDECURSOR = "\033[?25l"
	// Show cursor
	SHOWCURSOR = "\033[?25h"
)

const (
	MIN  = 60
	HOUR = 60 * MIN
	DAY  = 24 * HOUR
)

const (
	MiB = 1024 * 1024
)