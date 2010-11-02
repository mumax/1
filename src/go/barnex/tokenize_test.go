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
	
}

type ErrHandler struct{

}

func (e ErrHandler) Error(pos token.Position, msg string){
  fmt.Println(pos, msg)
}