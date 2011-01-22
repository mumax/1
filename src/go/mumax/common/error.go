//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

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

// TODO: rename CheckErr
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
