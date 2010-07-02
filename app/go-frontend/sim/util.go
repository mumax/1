package sim

import(
  "log"
  "os"
  "fmt"
)

// crashes if the test is false
func assert(test bool){
  if !test{
    log.Crash("assertion failed");
  }
}

// puts a 3 in front of the array
func Size4D(size3D []int) []int{
  assert(len(size3D) == 3)
  size4D := make([]int, 4)
  size4D[0] = 3
  for i:=range size3D{
    size4D[i+1] = size3D[i]
  }
  return size4D
}

// removes the 3 in front of the array
func Size3D(size4D []int) []int{
  assert(len(size4D) == 4)
  assert(size4D[0] == 3)
  size3D := make([]int, 3)
  for i:=range size3D{
    size3D[i] = size4D[i+1]
  }
  return size3D
}

var verbosity int = 3

func Debug(msg string){
  if verbosity > 0 {
    fmt.Println(os.Stderr, msg)
  }
}

func Debugv(msg string){
  if verbosity > 1{
    Debug(msg)
  }
}

func Debugvv(msg string){
    if verbosity > 2{
    Debug(msg)
  }
}