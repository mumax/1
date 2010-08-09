package refsh

import (
	"testing"
	"os"
	"fmt"
)

func TestTokenizer(test *testing.T) {
	in, err := os.Open("test.in", os.O_RDONLY, 0666)
	if err != nil {
		test.Fail()
		return
	}
	
	for line, eof := ReadNonemptyLine(in); !eof; line, eof = ReadNonemptyLine(in){
		fmt.Println(line)
	}
}
