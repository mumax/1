package refsh

import (
	"testing"
	//"os"
	"fmt"
)

func TestRefsh(test *testing.T) {
	//   in, err := os.Open("test.in", os.O_RDONLY, 0666)
	//   if err != nil{ test.Fail(); return }
	//

	refsh := NewRefsh()
	refsh.AddFunc("hello", Hello)
	refsh.AddFunc("echo", Echo)

	refsh.AddMethod("method", &St{1}, "Method")

	refsh.CrashOnError = false
	refsh.Interactive()

}

type St struct {
	It int
}

func (s *St) Method() int {
	return s.It
}

func Hello() {
	fmt.Println("Hello world!")
}

func Echo(i int) {
	fmt.Println(i)
}
