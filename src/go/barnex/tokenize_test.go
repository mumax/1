package barnex

import (
	"testing"
	"fmt"
)


func TestTokenize(t *testing.T) {
  node := Tokenize("test.txt")
  for node != nil{
    fmt.Println(node)
    node = node.Next
  }
}
