package core

import( 
  "testing";
  "os";
  "fmt";
  . "math";
)

func TestConv(t *testing.T){
  N0 := 8; N1 := 8; N2 := 2;
  size := []int{N0, N1, N2};
  size3D := []int{3, N0, N1, N2};
  
  kernel := NewTensor5([]int{3, 3, N0, N1, N2});
  for i:=0; i<3; i++{
    kernel.Array()[i][i][0][0][0] = 1.;
  }

  conv := NewConvPlan(size, kernel);

  m := NewTensor4(size3D);
  h := NewTensor4(size3D);

  for i:=range(m.List()){
    m.List()[i] = 1.;
  }

  ExecuteConv(conv, m, h);
  fmt.Println("m");
  Format(os.Stdout, m);
  fmt.Println("h");
  Format(os.Stdout, h);

  for i:=range(m.List()){
    if Fabs(float64(m.List()[i] - h.List()[i])) > 1E-4 { fmt.Println("err: ", Fabs(float64(m.List()[i] - h.List()[i]))); t.Fail()}
  }
}
