package sim

import( . "testing";
	. "../tensor";
	//. "../fft";
	//"fmt";
	//. "os";
	//"rand";
	//. "math";
)

func TestSim(t *T){

}

func TestField(t *T){
   const(N0 = 32; 
       N1 = 32;
       N2 = 1;
  );
  size := []int{N0, N1, N2};
  size3D := []int{3, N0, N1, N2};
  field := NewFieldPlan(size, []float{5E-9, 5E-9, 5E-9}, 800E3, 1.3E-11);
  //s.PrintInfo();
 
  M, H := NewTensor4(size3D), NewTensor4(size3D);
  M.Array()[X][N0/2][N1/2][N2/2] = 1.;
  field.Execute(M, H);
  PrintVectors(FOpen("h"), Normalize(H, 0));
}

func BenchmarkProblem4(b *B){
 const(N0 = 32; 
       N1 = 8;
       N2 = 1;
  );

  size := []int{N0, N1, N2};
  size3D := []int{3, N0, N1, N2};
  s := NewFieldPlan(size, []float{5E-9, 5E-9, 5E-9}, 800E3, 1.3E-11);
  s.PrintInfo();
 
  M := NewTensor4(size3D);
  euler := NewEuler(M, s, 0.001);
  
  SetAll(M.Component(X), 1.);
  SetAll(M.Component(Y), 1.);
  
  t:=0;
  for i:=0; i<1000; i++{
      euler.Step();
      t++;
    }
   PrintVectors(FOpen("probl4"), M);
}

func BenchmarkConv(b *B){
//   b.StopTimer();
//    const(N0 = 128; 
//         N1 = 128;
// 	N2 = 1
//   );
// 
//   size := []int{N0, N1, N2};
//   size3D := []int{3, N0, N1, N2};
// 
//   m := NewTensor4(size3D);
// 
//   h := NewTensor4(size3D);
//   K := DipoleKernel(size);
//   conv := NewConvPlan(m, h, K); // destroys data
//   
//   m.Array()[0][N0/2][N1/2][0] = 1.; // unit magnetization in center
// 
//   b.StartTimer();
//   for i:=0; i<1; i++{
//   conv.Execute();
//  }
//    PrintVectors(FOpen("/home/arne/Desktop/Kx"), Normalize(Slice(K, 0, X), 0));
//    PrintVectors(FOpen("/home/arne/Desktop/Ky"), Normalize(Slice(K, 0, Y), 0));
//    PrintVectors(FOpen("/home/arne/Desktop/Kz"), Normalize(Slice(K, 0, Z), 0));
//   
//   PrintVectors(FOpen("/home/arne/Desktop/m"), m);
//   PrintVectors(FOpen("/home/arne/Desktop/h"), Normalize(h, 0));
}


