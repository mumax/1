package sim

// This file implements functions for writing to stdout/stderr
// and simultaneously to a log file. (Unix "tee" functionality)

func (s *Sim) Println