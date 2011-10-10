//  This file is part of MuMax, a high-performance micromagnetic simulator
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
	. "mumax/common"
	"flag"
	"os"
	"exec"
	"path"
	"fmt"
)

const VERSION = 2495

const WELCOME = `
  MuMax 1.0.2503 beta
  (c) Arne Vansteenkiste & Ben Van de Wiele,
      DyNaMat/EELAB Ghent University.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

`

// Recognized input file extensions
var known_extensions []string = []string{".in", ".py"}

// returns true if the extension (e.g. ".in") is recognized
func is_known_extension(ext string) bool {
	for _, e := range known_extensions {
		if e == ext {
			return true
		}
	}
	return false
}

// returns true if the filename (e.g. "file.in") has a recognized extension
func has_known_extension(filename string) bool {
	return is_known_extension(path.Ext(filename))
}


// Start a mumax/python/... slave subprocess and tee its output
func main_master() {

	shouldupgrade := false
	if *flag_checkversion {
		CheckVersion("http://dynamag.ugent.be/mumax/latest-version.txt", VERSION)
	}

	if !*silent {
		fmt.Println(WELCOME)
		if shouldupgrade {
			fmt.Println("A newer version of MuMax is available")
		}
		PrintInfo()
	}

	if flag.NArg() == 0 {
		NoInputFiles()
		os.Exit(-1)
	}

	// Process all input files
	for i := 0; i < flag.NArg(); i++ {
		infile := flag.Arg(i)
		extension := path.Ext(infile)
		switch extension {
		default:
			UnknownFileFormat(extension)
			os.Exit(ERR_UNKNOWN_FILE_FORMAT)
		case ".in":
			main_raw_input(infile)
		case ".py":
			main_python(infile)
		}
	}
}

// Main for raw input ".in" files
func main_raw_input(infile string) {

	args := passthrough_cli_args()
	args = append(args, "--slave", infile)
	cmd, err := subprocess(os.Getenv(SIMROOT)+"/"+SIMCOMMAND, args, exec.DevNull, exec.Pipe, exec.Pipe)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(ERR_SUBPROCESS)
	} else {
		if !*silent {
			fmt.Println("Child process PID ", cmd.Process.Pid)
		}
		go Pipe(cmd.Stdout, os.Stdout) // TODO: logging etc
		go Pipe(cmd.Stderr, os.Stderr)
		_, errwait := cmd.Wait(0) // Wait for exit
		if errwait != nil {
			fmt.Fprintln(os.Stderr, err)
			os.Exit(ERR_SUBPROCESS)
		}
	}
}
