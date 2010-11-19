package barnex

import (
	"testing"
	"fmt"
)


func TestTokenize(t *testing.T) {
	node, _ := Tokenize("test.txt")
	for node != nil {
		fmt.Println(node)
		node = node.Next
	}

	Parse("test.txt")
}
