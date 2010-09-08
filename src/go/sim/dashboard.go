package sim

import (
	"fmt"
	"os"
)


func updateDashboard(sim *Sim) {
  savePos()
  fmt.Print("step:", sim.steps)
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

const (
	// Save cursor position
	SAVEPOS = "\033[s"

	// Erase rest of line
	ERASE = "\033[K"

	// Restore cursor position
	RESTOREPOS = "\033[u"
)
