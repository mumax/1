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
type Timer map[string]*Stopwatch

func NewTimer() Timer {
	return Timer(make(map[string]*Stopwatch))
}

func (t Timer) Start(tag string) {
	s, ok := t[tag]
	if !ok {
		s = new(Stopwatch)
		t[tag] = s
	}
	if s.start != 0 {
		panic("Timer.Start(" + tag + "): already running")
	}
	s.start = time.Nanoseconds()
}

func (t Timer) Stop(tag string) {
	s, ok := t[tag]
	if !ok {
		panic("Timer.Stop(" + tag + "): was not started")
	}
	s.total += (time.Nanoseconds() - s.start)
	s.invocations++
	s.start = 0
}

func (t Timer) PrintTimer(out io.Writer) {
	for tag, s := range t {
		fmt.Fprintln(out, tag, ":", s.total/1000000, "ms", "(", float64(s.total)/float64(1000000*int64(s.invocations)), "ms/invocation)")
	}
}

// INTERNAL
// Tracks the timing for one tag.
type Stopwatch struct {
	start, total int64 // start == 0 indicates the stopwatch is not running.
	invocations  int
}
