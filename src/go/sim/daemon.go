
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"flag"
	"os"
	"fmt"
	"strings"
	"time"
	"exec"
)

const (
	DAEMON_WATCHTIME = 2 // search for new input files every X s
	DAEMON_PREFIX    = BOLD + "[daemon] "
	DAEMON_SUFFIX    = RESET
)

const SIMCOMMAND = "bin/simulate"

func DaemonMain() {
	sleeping := false

	fmt.Println(DAEMON_PREFIX, "Input files should end with .in and the corresponding .out directory should not yet exist.", DAEMON_SUFFIX)

	// TODO check that the watchdirs do not end with a slash
	watchdirs := make([]string, flag.NArg())
	for i := range watchdirs {
		watchdirs[i] = flag.Arg(i)
	}
	// if no watch dir is specified, look in the current directory
	if len(watchdirs) == 0 {
		watchdirs = []string{"."}
	}
	for {
		infile := findInputFileAll(watchdirs)
		if infile == "" {
			if !sleeping {
				fmt.Println(DAEMON_PREFIX, "Looking for new input files every ", DAEMON_WATCHTIME, " seconds", DAEMON_SUFFIX)
			}
			sleeping = true
			time.Sleep(DAEMON_WATCHTIME * 1E9)
		} else {
			sleeping = false
			daemon_startsim(infile)
		}
	}
}

func daemon_startsim(file string) {
	fmt.Println(DAEMON_PREFIX, "Starting simulation: ", file, DAEMON_SUFFIX)

	// We try to create the output directory before starting the simulation.
	// This acts as a synchronization mechanism between multiple daemons:
	// should another daemon already have started this simulation in the meanwhile,
	// then this directory exists and we should abort.
	outfile := removeExtension(file) + ".out"
	err := os.Mkdir(outfile, 0777)
	// if the directory already exists, then another daemon had already started the simulation in the meanwhile
	// TODO: we should check if the error really is a "file exists"
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return
	}

	wd, err3 := os.Getwd()
	if err3 != nil {
		panic(err3)
	}

	cmdstr := os.Getenv("SIMROOT") + "/" + SIMCOMMAND
	args := []string{"simulate", file} // aparently argument 1, not argument 0 is the first real argument, we pass "simulate" as a dummy argument (probably program name)
	//fmt.Println("exec ", cmdstr, args)
	cmd, err2 := exec.Run(cmdstr, args, os.Environ(), wd, exec.PassThrough, exec.PassThrough, exec.MergeWithStdout)
	if err2 != nil {
		fmt.Fprintln(os.Stderr, err2)
	} else {
		_, err4 := cmd.Wait(0)
		if err4 != nil {
			fmt.Fprintln(os.Stderr, err4)
		} else {
			fmt.Println(DAEMON_PREFIX, "Finished simulation ", file, DAEMON_SUFFIX)
		}
	}
}


// Searches for a pending input file in all the given directories.
// Looks for a file ending in ".in" for which no corresponding
// ".out" file exists yet.
// Returns an empty string when no suitable input file is present
// in any of the directories.
// TODO: Should we look recursively?
//
func findInputFileAll(dirs []string) string {
	for _, dir := range dirs {
		file := findInputFile(dir)
		if file != "" {
			return file
		}
	}
	return "" // nothing found
}

// Searches for a pending input file in the given directory.
// Looks for a file ending in ".in" for which no corresponding
// ".out" file exists yet.
// Returns an empty string when no suitable input file is present.
//
func findInputFile(dir string) string {
	d, err := os.Open(dir, os.O_RDONLY, 0666)
	// if we can not read a directory, we should not necessarily crash,
	// instead report it and go on so other directories can still be searched.
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return ""
	}
	defer d.Close()

	// loop over Readdirnames(), all files in the directory
	for filenames, err2 := d.Readdirnames(1); err2 == nil; filenames, err2 = d.Readdirnames(1) {
		if len(filenames) == 0 { //means we reached the end of the files
			return ""
		}
		file := filenames[0]
		if strings.HasSuffix(file, ".in") && !fileExists(dir+"/"+removeExtension(file)+".out") {
			return dir + "/" + file
		}
	}

	// nothing found
	return ""
}

// checks if the file exists
func fileExists(file string) bool {
	f, err := os.Open(file, os.O_RDONLY, 0666)
	if err != nil {
		return false
	}
	f.Close()
	return true
}

// --- Daemons are characters in Greek mythology,
//     some of whom handled tasks that the gods
//     could not be bothered with.
