package tensor

import "log"

//TODO: move to util?

/** Crashes the program when the test is false. */
func assert(test bool) {
	if !test {
		log.Crash("Assertion failed")
	}
}

/** Crashes the program with an error message when the test is false. */
func assertMsg(test bool, msg string) {
	if !test {
		log.Crash(msg)
	}
}
