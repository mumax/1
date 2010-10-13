package tensor2

import (
	"testing"
	"os"
)

func TestMisc(test *testing.T) {
	t := NewT4([]int{1, 2, 3, 4})
	WriteHeader(os.Stdout, t)
	WriteDataBinary(os.Stdout, t)
}
