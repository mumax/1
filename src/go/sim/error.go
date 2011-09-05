package sim

import (
	"os"
	"fmt"
)

// We define different error types so a recover() after
// panic() can determine (with a type assertion)
// what kind of error happenend. 
// Only a Bug error causes a bugreport and stackdump,
// other errors are the user's fault and do not trigger
// a stackdump.

// The input file contains illegal input
type InputErr string

// A file could not be read/written
type IOErr string

// An unexpected error occured which sould be reported
type Bug string


func Check(err os.Error, code int) {
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(code)
	}
}

// Exit error code
const (
	ERR_INPUT               = 1
	ERR_IO                  = 2
	ERR_UNKNOWN_FILE_FORMAT = 3
	ERR_SUBPROCESS          = 4

	ERR_BUG = 255
)
