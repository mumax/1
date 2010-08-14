package refsh

import (
    . "reflect"
	"testing"
	//"os"
	"fmt"
)

func TestRefsh(test *testing.T) {
	//   in, err := os.Open("test.in", os.O_RDONLY, 0666)
	//   if err != nil{ test.Fail(); return }
	//

	refsh := NewRefsh()
	refsh.Add("hello", Hello)
	refsh.Add("echo", Echo)
	var s interface{}
	s = S{1}
	var iv *InterfaceValue
	iv = &NewValue(s)
	refsh.Add("method", .Method(0))
	refsh.CrashOnError = false
	refsh.Interactive()

}

type S struct{
  i int
}

func (s S) Method() int{
  return s.i
}

func Hello() {
	fmt.Println("Hello world!")
}

func Echo(i int) {
	fmt.Println(i)
}
