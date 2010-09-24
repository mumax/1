package main

import (
	"sim"
)

// Wrapper for sim.Main.
// This is a bit silly but sim.Main() is in package sim (not package main)
// so it cannot be compiled to an executable directly.
func main() {
	sim.Main()
}
