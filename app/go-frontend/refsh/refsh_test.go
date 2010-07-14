package refsh

import(
  "testing"
  "os"
  "fmt"
)

func TestRefsh(test *testing.T){
  in, err := os.Open("test.in", os.O_RDONLY, 0666)
  if err != nil{ test.Fail(); return }
  
  refsh := NewRefsh()
  refsh.Add("hello", Hello)
  refsh.Add("echo", Echo)
  refsh.Parse(in)
  
}


func Hello(){
  fmt.Println("Hello world!")
}

func Echo(i int){
  fmt.Println(i)
}




