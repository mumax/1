package refsh

import (
	"io"
	"container/vector"
	"fmt"
	"os"
)


type Tokenizer struct{
  io.Reader
}
// func NewTokenizer(in io.Reader) *Tokenizer{
//   return &Tokenizer{in, vector.StringVector(make([]string, 10))}
// }

func (t *Tokenizer) ReadChar() int {
	buffer := [1]byte{}
	switch nr, err := t.Read(buffer[0:]); true {
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

func (t *Tokenizer) ReadLine() []string {
    words_arr := [10]string{}
	words := vector.StringVector(words_arr[0:0])
	currword := ""
	for {
		char := t.ReadChar()
		//fmt.Println("char:", char)
		if isEndline(char) {
            if currword != ""{
              words.Push(currword)
              currword = ""
            }
			return []string(words)
		}
		
        if isWhitespace(char) && currword != "" {
            words.Push(currword)
            currword = ""
        } // whitespace && currword == "": ignore whitespace
        
        if isCharacter(char){
          currword += string(char)
        }
	}

	//not reached
	return []string(words)
}

func isEndline(char int) bool {
	if char == int('\n') || char == -1 || char == int(';'){
		return true
	}
	return false
}

func isWhitespace(char int) bool {
	if char == int(' ') || char == int('\t') || char == int(':'){
		return true
	}
	return false
}

func isCharacter(char int) bool{
  return !isEndline(char) && !isWhitespace(char)
}
