package sim

import (
	"fmt"
	"os"
	"time"
)

var lastDashUpdate int64 = 0
var UpdateDashboardEvery int64 = 100 * 1000 * 1000 // in ns

func updateDashboard(sim *Sim) {

  T := sim.UnitTime()
    
	nanotime := time.Nanoseconds()
	if (nanotime - lastDashUpdate) < UpdateDashboardEvery {
		return // too soon to update display yet
	}
	lastDashUpdate = nanotime

	savePos()
	//fmt.Print(HIDECURSOR)
	// Walltime
	time := time.Seconds() - sim.starttime
	fmt.Printf(
		BOLD+"running:"+RESET+"%3dd:%02dh:%02dm:%02ds\n",
		time/DAY, (time/HOUR)%24, (time/MIN)%60, time%60)

	// Time stepping
	fmt.Printf(
		BOLD+"step: "+RESET+"%-11d "+
			BOLD+"time: "+RESET+"%.4es      "+
			BOLD+"Δt:   "+RESET+" %.3es",
		sim.steps, float(sim.time)*T, sim.dt*T)
	erase()
	fmt.Println()

	// Conditions
	fmt.Printf(
		BOLD+"B:    "+RESET+"(%.3e, %.3e, %.3e)T",
		sim.hext[0], sim.hext[1], sim.hext[2])
	erase()
	fmt.Println()

	restorePos()
}


func savePos() {
	fmt.Fprintf(os.Stdout, SAVEPOS)
}

func erase() {
	fmt.Fprintf(os.Stdout, ERASE)
}

func restorePos() {
	fmt.Fprintf(os.Stdout, RESTOREPOS)
}

// ANSI escape sequences
const (
	// Save cursor position
	SAVEPOS = "\033[s"
	// Erase rest of line
	ERASE = "\033[K"
	// Restore cursor position
	RESTOREPOS = "\033[u"
	// Hide cursor
	HIDECURSOR = "\033[?25l"
	// Reset font
	RESET = "\033[0m"
	// Bold
	BOLD = "\033[1m"
)

const (
	MIN  = 60
	HOUR = 60 * MIN
	DAY  = 24 * HOUR
)
