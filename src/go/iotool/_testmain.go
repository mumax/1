package main

import "iotools"
import "testing"

var tests = []testing.Test {
	testing.Test{ "refsh.TestReadline", refsh.TestReadline },
}
var benchmarks = []testing.Benchmark {
}

func main() {
	testing.Main(tests);
	testing.RunBenchmarks(benchmarks)
}
