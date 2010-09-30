//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package refsh

import (
	"flag"
	"fmt"
	"os"
	"strings"
)

/**
 * Parses the command line flags with the following layout:
 * --command0 arg0 arg1 --command1 arg0 --command2 ...
 * Returns an array of commands and args. command[i] has args args[i].
 */
func ParseFlags() ([]string, [][]string) {
	ncommands := 0
	for i := 0; i < flag.NArg(); i++ {
		if strings.HasPrefix(flag.Arg(i), "--") {
			ncommands++
		}
	}

	commands := make([]string, ncommands)
	args := make([][]string, ncommands)

	{
		command := 0
		i := 0
		for i < flag.NArg() {
			//assert(strings.HasPrefix(flag.Arg(i), "--"));
			if !strings.HasPrefix(flag.Arg(i), "--") {
				fmt.Println("Expecting a command starting with --")
				os.Exit(-2)
			}
			commands[command] = flag.Arg(i)
			nargs := 0
			i++
			for i < flag.NArg() && !strings.HasPrefix(flag.Arg(i), "--") {
				nargs++
				i++
			}
			args[command] = make([]string, nargs)
			command++
		}
	}

	{
		command := 0
		i := 0
		for i < flag.NArg() {
			//assert(strings.HasPrefix(flag.Arg(i), "--"));
			commands[command] = flag.Arg(i)[2:len(flag.Arg(i))]
			nargs := 0
			i++
			for i < flag.NArg() && !strings.HasPrefix(flag.Arg(i), "--") {
				args[command][nargs] = flag.Arg(i)
				nargs++
				i++
			}
			command++
		}
	}
	return commands, args
}
