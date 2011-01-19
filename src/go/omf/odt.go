//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package omf


// This file implements output in OOMMF's .odt table format
// example:
// table := omf.NewTabWriter(io_writer)
// table.AddColumn("Mx", "A/m")
// table.Print(0.95)
// table.Close()


import (
	"fmt"
	"io"
	"bufio"
	"tabwriter"
)


type TabWriter struct {
	out       io.Writer
	bufout    *bufio.Writer
	tabout    *tabwriter.Writer
	Title     string
	columns   []string
	units     []string
	initiated bool
	closed    bool
	colcount  int
}


func NewTabWriter(out io.Writer) *TabWriter {
	t := new(TabWriter)
	t.out = out
	t.bufout = bufio.NewWriter(t.out)
	t.columns = []string{}
	t.units = []string{}
	return t
}


func (t *TabWriter) AddColumn(colname, unit string) {
	if t.initiated {
		panic("Can not add column when omf.TabWriter is already open")
	}
	t.columns = append(t.columns, colname)
	t.units = append(t.units, unit)
}


func (t *TabWriter) Print(v ...interface{}) {
	if !t.initiated {
		t.open()
	}
	for _, val := range v {
		fmt.Fprint(t.tabout, val, " \t")
		t.colcount++
		if t.colcount == len(t.columns) {
			fmt.Fprintln(t.tabout)
			t.colcount = 0
		}
	}
}


func (t *TabWriter) Close() {
	fmt.Fprintln(t.tabout, "# Table End")
	t.tabout.Flush()
	t.bufout.Flush()
	if closer := t.out.(io.Closer); closer != nil {
		closer.Close()
	}
}

const COL_WIDTH = 20

func (t *TabWriter) open() {
	t.tabout = tabwriter.NewWriter(t.bufout, COL_WIDTH, 4, 0, ' ', 0)
	out := t.tabout
	fmt.Fprintln(out, "# ODT 1.0")
	fmt.Fprintln(out, "# Table Start")
	fmt.Fprintln(out, "# Title: ", t.Title)

	fmt.Fprint(out, "# Units:")
	for _, u := range t.units {
		fmt.Fprint(out, "{", u, "} \t")
	}
	fmt.Fprintln(out)

	fmt.Fprint(out, "# Columns:")
	for _, c := range t.columns {
		fmt.Fprint(out, c, " \t")
	}
	fmt.Fprintln(out)
	t.initiated = true
}
