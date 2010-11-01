package barnex

import (
	"testing"
	"go/scanner"
	"go/token"
	"fmt"
)


func TestTokenize(t *testing.T) {
	var s scanner.Scanner
	handler := ErrHandler{}
	s.Init(fname, souce, handler, 0)
	
}

type ErrHandler struct{

}

func (e *ErrHandler) Error(pos token.Position, msg string){
  fmt.Println(pos, msg)
}