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
	tokenizer := &Tokenizer{in}
	for i := 0; i < 10; i++ {
		words := tokenizer.ReadLine()
		fmt.Println(words)
	}
}
