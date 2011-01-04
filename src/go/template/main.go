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
	"iotool"
	"os"
	"strings"
	"strconv"
)


func main() {
	if flag.NArg() == 0 {
		Error("No input files.")
	}

	file := flag.Arg(flag.NArg() - 1)   // last flag is the template filename...
	bytes, err := ioutil.ReadFile(file) // ...read it fully
	content := string(bytes)
	if err != nil {
		panic(err)
	}

	// list of "documents" that will be output,
	// {key}'s will gradually be replaced by their values:
	// for value by which a key is being replaced,
	// a new document is appended to the list.
	// E.g. key=val1,val2 appends 2 documents with key
	// replaced by val1 and val2, respectively.
	// the original document is then replaced by nil.
	docs := []*Document{&Document{content, file}}

	for f := 0; f < flag.NArg()-1; f++ {

		// Split the flag in a key/value pair
		flag := flag.Arg(f)
		split := strings.Split(flag, "=", 2)
		if len(split) != 2 {
			Error("Syntax error: expecting \"key=value\"")
		}
		key, val := split[0], split[1]

		// Generate a sequence of values,
		// depending on whether the value has a comma- or colon-separted layout
		values := GenerateValues(val)

		len_docs := len(docs) // no need to iterate over those that will be added by this loop
		for d := 0; d < len_docs; d++ {
			doc := docs[d]
			if doc != nil {
				docs[d] = nil
				for _,v := range values {
					docs = append(docs, doc.Replace(key, v))
				}
			}
		}
	}

	for _, d := range docs {
		if d != nil {
			if strings.Contains(d.content, "{") {
				Error("Not all {key}'s were specified.")
				// TODO: it might be nice to show which ones...
			}
			out := iotool.MustOpenWRONLY(d.name + ".in")
			defer out.Close()
			out.Write([]byte(d.content))
		}
	}

}

// Generate a sequence of values,
// depending on whether the value has a comma- or colon-separted layout.
// E.g.:
// 1,2,3 -> {1, 2, 3}
// 1:5 -> {1, 2, 3, 4}
// 0:10:2 -> {0, 2, 4, 6, 8}
func GenerateValues(flag string) []string{
  
  // comma separated list
  if strings.Contains(flag, ","){
    values := strings.Split(flag, ",", -1)
    return values
  }

  // range statement
  if strings.Contains(flag, ":"){
    steps := strings.Split(flag, ":", -1)
    switch len(steps){
      default: Error("Sytax error: expecting \"start:stop\" or \"start:stop:step\" :", flag)
      case 2: return Range(steps[0], steps[1], "1")
      case 3: return Range(steps[0], steps[1], steps[2])
    }
  }

  // just one value
  return []string{flag}
}


func Range(start, stop, step string) []string{
  rnge := []string{}

  min, max, delta := Atof(start), Atof(stop), Atof(step)

  for i:=min; i<max; i+=delta{
    rnge = append(rnge, fmt.Sprint(float(i)))
  }

  return rnge
}


func Atof(a string) float64{
  f, err := strconv.Atof64(a)
  if err != nil{
    Error("Parsing " + a + ": " + err.String())
  }
  return f
}


type Document struct {
	content string
	name    string
}

func (d *Document) Replace(key, val string) *Document {
	if !strings.Contains(d.content, key) {
		Error("Template file does not contain key: {" + key + "}")
	}
	d2 := new(Document)
	d2.content = strings.Replace(d.content, "{"+key+"}", val, -1)
	d2.name = d.name + "_" + key + "=" + val
	return d2
}

func Error(msg ...interface{}) {
	fmt.Fprintln(os.Stderr, msg...)
	fmt.Fprintln(os.Stderr, USAGE)
	os.Exit(-1)
}

const USAGE = `
Usage: template file should contain {key} statements, where "key" can be replaced by any identifier.
template key=value1,value2,... file.in  Creates files where {key} is replaced by each of the values.
template key=start:stop file.in         Replaces key by all integers between start and stop (exclusive).
template key=start:stop:step file.in    As above but with a step different from 1.
template key1=... key2=...              Multiple keys may be specified.
Output files are given automaticially generated names.
`
