package sim

import (
	"flag"
	"os"
	"fmt"
	"strings"
)


func DaemonMain() {
	// TODO check that the watchdirs do not end with a slash
	watchdirs := make([]string, flag.NArg())
	for i := range watchdirs {
		watchdirs[i] = flag.Arg(i)
	}
	// if no watch dir is specified, look in the current directory
	if len(watchdirs) == 0 {
		watchdirs = []string{"."}
	}

	infile := findInputFile(watchdirs[0])
	fmt.Println(infile)
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
		Debugv(err)
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
			return file
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
