package fft

import( . "testing";
	. "../tensor";
	//. "os";
	"rand";
	. "math";
)

func TestConv(t *T){
 const(N0 = 8; 
        N1 = 8;
	N2 = 1
  );

  size := []int{N0, N1, N2};
  size3D := []int{3, N0, N1, N2};

  // make some random data
  m := NewTensor4(size3D);
  for i:=range(m.List()){
    m.List()[i] = rand.Float()+0.01; // so it's not zero
  }
  
  // copy the original data for later comparison
  orig := NewTensor4(m.Size());
  CopyTo(m, orig);

  //PrintTensor(Stdout, m.Component(0));

  // convolve with unit tensor
  h := NewTensor4(size3D);
  K := UnitKernel(size);
  conv := NewConvPlan(m.Component(0).Size(), K); // destroys data
  conv.Execute(m, h);

//   PrintVectors(FOpen("/home/arne/Desktop/Kx.txt"), Normalize(Slice(K, 0, X), 0));
//   PrintVectors(FOpen("/home/arne/Desktop/Ky.txt"), Normalize(Slice(K, 0, Y), 0));
//   PrintVectors(FOpen("/home/arne/Desktop/Kz.txt"), Normalize(Slice(K, 0, Z), 0));
  
   
// 
//   m.Component(0).Set(N0/2, N1/2, N2/2, 1.);
//   //m.Component(1).Set(N0/2, N1/2, N2/2, 3.);
//   //m.Component(2).Set(0.);
//   PrintVectors(FOpen("/home/arne/Desktop/m.txt"), m.As4Tensor());
// 
   
   //PrintTensor(Stdout, h.Component(0));
   //PrintTensor(Stdout, orig.Component(0));
//   
//   PrintVectors(FOpen("/home/arne/Desktop/h.txt"), Normalize(h.As4Tensor(), 0));

  // Test if data remained unchanged by unit convolution.
  ma:=h.List();
  for i:= range(ma){
    // also, we do not allow zero's, as it is too dangerous to be comparing to zero arrays by accident.
    if Fabs(float64(ma[i] - orig.List()[i])) > 0.000001  || ma[i] == 0.{ t.Fail(); }
  }

 }


func UnitKernel(unpaddedsize []int) StoredTensor{
  size := PadSize(unpaddedsize);
  k := NewTensor5([]int{3, 3, size[0], size[1], size[2]});
  for c:=0; c<3; c++{
	k.Array()[c][c][0][0][0] = 1.;
  }
  return k;
}
