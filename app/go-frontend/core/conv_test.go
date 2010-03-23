package core

import( 
  . "testing";
)

func TestConv(t *T){
  N0 := 8; N1 := 8; N2 := 2;
  size := []int{N0, N1, N2};
  size3D := []int{3, N0, N1, N2};
  
  kernel := NewTensor5([]int{3, 3, N0, N1, N2});

  conv := NewConvPlan(size, kernel);
  m := NewTensor4(size3D);
  h := NewTensor4(size3D);

  ExecuteConv(conv, m, h);
  
}
