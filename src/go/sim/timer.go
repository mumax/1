//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
	"time"
	"io"
	"fmt"
)

// A timer for timing code execution
// and counting number of invocations.
// A "tag"-string is passed to each start/stop
// to identify what is being timed
//
// Example Usage:
// t := NewTimer()
// t.Start("mainloop")
// mainloop()
// t.Stop("mainloop")
// t.Start("other loop")
// otherloop()
// t.Stop("other loop")
// t.PrintTimer(os.Stdout)
//
type Timer struct{
  clocks map[string]*Stopwatch
  current *Stopwatch
}

func NewTimer() Timer {
	return Timer{make(map[string]*Stopwatch), nil}
}

func (t *Timer) Start(tag string) {
  if t.current != nil{
    panic(Bug("Timer.Start(" + tag + "): already running"))
  }
	s, ok := t.clocks[tag]
	if !ok {
		s = new(Stopwatch)
		t.clocks[tag] = s
	}
	if s.start != 0 {
		panic(Bug("Timer.Start(" + tag + "): already running"))
	}
	s.start = time.Nanoseconds()
	t.current = s
}

func (t *Timer) Stop() {
	s := t.current
	if s == nil {
		panic(Bug("Timer.Stop(): was not started"))
	}
	s.total += (time.Nanoseconds() - s.start)
	s.invocations++
	s.start = 0
	t.current = nil
}

func (t *Timer) PrintTimer(out io.Writer) {
	for tag, s := range t.clocks {
		fmt.Fprintln(out, tag, ":", float(s.total)/1000000000, "s", "(", float64(s.total)/float64(1000000*int64(s.invocations)), "ms/invocation)")
	}
}

// INTERNAL
// Tracks the timing for one tag.
type Stopwatch struct {
	start, total int64 // start == 0 indicates the stopwatch is not running.
	invocations  int
}
