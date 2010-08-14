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

  s := &St{1}
  t := Typeof(s)
  m := t.Method(0)
  f := m.Func
  fmt.Println(f)
	refsh.AddMethod("method", f)
	
	refsh.CrashOnError = false
	refsh.Interactive()

}

type St struct{
  It int
}

func (s *St) Method() int{
  return s.It
}

func Hello() {
	fmt.Println("Hello world!")
}

func Echo(i int) {
	fmt.Println(i)
}
