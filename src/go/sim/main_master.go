//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements the main() for mumax running in master mode.
// Master mode starts sub-processes:
// 	* A slave mumax sub-process that will do the actual simulation.
//    Its stderr is tee'ed to a log file and possible the terminal,
//	  making sure all output is logged even it crashes ungracefully.
//  * Possibly a python/java/... subprocess to interpret input files
//    written in those languages. The master connects them to the mumax
//    slave with pipes and makes sure their output gets logged as well. 
//
// TODO: master could have have the ability to automatically select between GPU/CPU
import (
	"flag"
	"os"
	"exec"
	"fmt"
)

const WELCOME = `
  MuMax 0.4.1747
  (c) Arne Vansteenkiste & Ben Van de Wiele,
      DyNaMat/EELAB UGent
  This version is meant for internal testing purposes only,
  please contact the authors if you like to distribute this program.
  
`
// Start a mumax/python/... slave subprocess and tee its output
func main_master() {

	if !*silent {
		fmt.Println(WELCOME)
	}

	if flag.NArg() == 0 {
		NoInputFiles()
		os.Exit(-1)
	}

	// Process all input files
	for i := 0; i < flag.NArg(); i++ {
		infile := flag.Arg(i)
		args := passthrough_cli_args()
		args = append(args, "--slave", infile)
		cmd, err := subprocess(os.Getenv(SIMROOT)+"/"+SIMCOMMAND, args, exec.DevNull, exec.Pipe, exec.Pipe)
		if err != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(-7)
		} else {
			fmt.Println("Child process PID ", cmd.Pid)
			go passtroughStdout(cmd.Stdout)
			_, errwait := cmd.Wait(0) // Wait for exit
			if errwait != nil {
				fmt.Fprintln(os.Stderr, err)
				os.Exit(-8)
			}
		}
	}
}
