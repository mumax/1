package refsh

import(
  "testing"
  "os"
)

func TestRefsh(test *testing.T){
  in, err := os.Open("test.in", os.O_RDONLY, 0666)
  if err != nil{ test.Fail(); return }
  
  refsh := NewRefsh()
  refsh.Parse(in)
  
}

