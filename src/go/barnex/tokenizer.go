package barnex

import (
	"go/scanner"
	"go/token"
	"io/ioutil"
	"fmt"
	"os"
)


func Tokenize(fname string) (start, stop *Node) {
	source, err := ioutil.ReadFile(fname)
	if err != nil {
		panic(err)
	}
	var s scanner.Scanner
	handler := ErrHandler{}
	s.Init(fname, source, handler, scanner.InsertSemis)

	start = NewRootNode()
	prev := start

	pos, tok, lit := s.Scan()
	for tok != token.EOF {
		node := NewNode(pos, tok, string(lit))
		prev.Append(node)
		pos, tok, lit = s.Scan()
		prev = node
	}

	stop = NewEOFNode()
	prev.Append(stop)

	return
}


type ErrHandler struct{}

func (e ErrHandler) Error(pos token.Position, msg string) {
	fmt.Fprintln(os.Stderr, pos, msg)
	panic(msg)
}
