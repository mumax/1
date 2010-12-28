//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package refsh


// Parses command line arguments to refsh commands,
// according to the conventions of the Mercury Project.

import (
	"flag"
// 	"fmt"
// 	"os"
	. "strings"
)


// Parses the command line flags with the following layout:
// --command0="arg0, arg1" --command1="arg0" --command2 file1 file2 ...
// Returns an array of commands and args. command[i] has args args[i].
// files contains the CLI aruments that do not start with --
func ParseFlags2() (commands []string, args [][]string, files []string) {

//   ncommands := 0
// 	nfiles := 0
// 	for i := 0; i < flag.NArg(); i++ {
// 		if strings.HasPrefix(flag.Arg(i), "--") {
// 			ncommands++
// 		} else {
// 			nfiles++
// 		}
// 	}

// 	commands = make([]string, ncommands)
// 	args = make([][]string, ncommands)
// 	files = make([]string, nfiles)

  for i:=0; i < flag.NArg() ; i++{
    if HasPrefix(flag.Arg(i), "--") {
      
      cmd, arg := parseFlag2(flag.Arg(i))
      commands = append(commands, cmd)
      args = append(args, arg)
      
    }else{
      files = append(files, flag.Arg(i))
    }
  }
  
// 	{
// 		command := 0
// 		file := 0
// 		i := 0
// 		for i < flag.NArg() {
// 			if !strings.HasPrefix(flag.Arg(i), "--") {
// 				files[file] = flag.Arg(i)
// 				file++
// 			} else {
// 
//         commands[command] = flag.Arg(i)
// 				nargs := 0
// 				i++
// 				for i < flag.NArg() && !strings.HasPrefix(flag.Arg(i), "--") {
// 					nargs++
// 					i++
// 				}
// 				args[command] = make([]string, nargs)
// 				command++
// 				
// 			}
// 		}
// 	}
// 
// 	{
// 		command := 0
// 		i := 0
// 		for i < flag.NArg() {
// 			//assert(strings.HasPrefix(flag.Arg(i), "--"));
// 			commands[command] = flag.Arg(i)[2:len(flag.Arg(i))]
// 			nargs := 0
// 			i++
// 			for i < flag.NArg() && !strings.HasPrefix(flag.Arg(i), "--") {
// 				args[command][nargs] = flag.Arg(i)
// 				nargs++
// 				i++
// 			}
// 			command++
// 		}
// 	}
	return
}

// splits "--command="arg1, arg2" into "command", {arg1, arg2}
func parseFlag2(flag string) (command string, args []string){
  assert(HasPrefix(flag, "--"))
  command = flag[2:]
  return
}

func assert(test bool){
  if !test{panic ("Assertion failed")}
}
