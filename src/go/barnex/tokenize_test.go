package barnex

import (
	"testing"
	"go/scanner"
	"go/token"
	"io/ioutil"
	"fmt"
)


func TestTokenize(t *testing.T) {
  fname := "test.txt"
  source, err := ioutil.ReadFile(fname)
  if err != nil{
    panic(err)
  }
	var s scanner.Scanner
	handler := ErrHandler{}
	s.Init(fname, source, handler, 0)

	pos, tok, lit := s.Scan()
  for tok != token.EOF{
    fmt.Println(pos, tok, string(lit))
    pos, tok, lit = s.Scan()
  }
}

type ErrHandler struct{

}

func (e ErrHandler) Error(pos token.Position, msg string){
  fmt.Println(pos, msg)
}