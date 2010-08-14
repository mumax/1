package refsh

import (
	"io"
	"container/vector"
	"fmt"
	"os"
)

func ReadChar(in io.Reader) int {
	buffer := [1]byte{}
	switch nr, err := in.Read(buffer[0:]); true {
	case nr < 0: // error
		fmt.Fprintln(os.Stderr, "read error:", err)
		os.Exit(1)
	case nr == 0: // eof
		return -1
	case nr > 0: // ok
		return int(buffer[0])
	}
	return 0 // never reached
}

func ReadCharNoComment(in io.Reader) int {
	char := ReadChar(in)
	if char == int('#') {
		for char != int('\n') && char != -1 {
			char = ReadChar(in)
		}
	}
	return char
}

func ReadLine(in io.Reader) (line []string, eof bool) {
	words_arr := [10]string{}
	words := vector.StringVector(words_arr[0:0])
	currword := ""
	for {
		char := ReadCharNoComment(in)

		if isEndline(char) {
			if currword != "" {
				words.Push(currword)
				currword = ""
			}
			eof = isEOF(char) && len(words) == 0
			line = []string(words)
			return
		}

		if isWhitespace(char) && currword != "" {
			words.Push(currword)
			currword = ""
		} // whitespace && currword == "": ignore whitespace

		if isCharacter(char) {
			currword += string(char)
		}
	}

	//not reached
	return
}

func ReadNonemptyLine(in io.Reader) (line []string, eof bool) {
	line, eof = ReadLine(in)
	for len(line) == 0 && !eof {
		line, eof = ReadLine(in)
	}
	return
}

func isEOF(char int) bool {
	return char == -1
}

func isEndline(char int) bool {
	if isEOF(char) || char == int('\n') || char == int(';') {
		return true
	}
	return false
}

func isWhitespace(char int) bool {
	if char == int(' ') || char == int('\t') || char == int(':') {
		return true
	}
	return false
}

func isCharacter(char int) bool {
	return !isEndline(char) && !isWhitespace(char)
}
