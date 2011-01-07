//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package main

import (
	"fmt"
	"flag"
	"io/ioutil"
	"os"
)


func main() {
	if flag.NArg() == 0 {
		fmt.Fprintln(os.Stderr, "No input files.")
		fmt.Fprintln(os.Stderr, USAGE)
		os.Exit(-1)
	}

	file := flag.Arg(flag.NArg() - 1)
	bytes, err := ioutil.ReadFile(file)
	content := string(bytes)

	if err != nil {
		panic(err)
	}

	//   for i:=0; i<flag.NArg()-1; i++{
	// 
	//   }

	fmt.Println(content)
}

const USAGE = `
Usage: template file should contain {key} statements, where "key" can be replaced by any identifier.

template key=value1,value2,... file.in  Creates files where {key} is replaced by each of the values.
template key=start:stop file.in         Replaces key by all integers between start and stop (exclusive).
template key=start:stop:step file.in    As above but with a step different from 1.
template key1=... key2=...              Multiple keys may be specified.

Output files are given automaticially generated names.
`
