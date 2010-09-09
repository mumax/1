package sim

import (
	"fmt"
	"os"
	"time"
)

const(
  MIN = 60
  HOUR = 60*MIN
  DAY = 24*HOUR
  )
  
func updateDashboard(sim *Sim) {
  savePos()
  fmt.Print(HIDECURSOR)
  // Walltime
  time := time.Seconds() - sim.starttime
  fmt.Printf(
    BOLD + "running:" + RESET + "%3dd:%02dh:%02dm:%02ds\n",
    time / DAY, (time / HOUR)%24, (time / MIN)%60, time % 60)
  
  // Time stepping
  fmt.Printf(
    BOLD + "step: " + RESET + "%-11d "  +
    BOLD + "time: " + RESET + "%.4es      " +
    BOLD + "Δt:   "   + RESET + " %.3es",
    sim.steps, sim.time, sim.dt)
  erase()
  fmt.Println()

  // Conditions
  fmt.Printf(
    BOLD + "B:    " + RESET + "(%.3e, %.3e, %.3e)T",
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
