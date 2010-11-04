package barnex

import (
	"go/scanner"
	"go/token"
	"io/ioutil"
	"fmt"
)


func Tokenize(fname string) *Node {
	source, err := ioutil.ReadFile(fname)
	if err != nil {
		panic(err)
	}
	var s scanner.Scanner
	handler := ErrHandler{}
	s.Init(fname, source, handler, scanner.InsertSemis)

	root := NewRootNode()
	prev := root

	pos, tok, lit := s.Scan()
	for tok != token.EOF {
		fmt.Println(pos, tok, lit)
		node := NewNode(pos, tok, string(lit))
		node.Prev = prev
		node.Prev.Next = node
		pos, tok, lit = s.Scan()
		prev = node
	}

	return root
}


type ErrHandler struct{}

func (e ErrHandler) Error(pos token.Position, msg string) {
	fmt.Println(pos, msg)
	panic(msg)
}
