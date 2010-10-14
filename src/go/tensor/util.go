package tensor


func Equals(a, b Interface) bool {
  if !EqualSize(a.Size(), b.Size()) {
    return false
  }
  listA, listB := a.List(), b.List()
  for i:= range listA{
    if listA[i] != listB[i] {return false}
  }
  return true
}



// Tests if both int slices are equal,
// in which case they represent equal Tensor sizes.

func EqualSize(a, b []int) bool {
  if len(a) != len(b) {
    return false
  } else {
    for i := range a {
      if a[i] != b[i] {
        return false
      }
    }
  }
  return true
}