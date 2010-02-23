package main
  
import( 
       . "../tensor";
       //. "../fft";
       . "../sim";
       "../libsim";
       . "fmt";
)

func main() {
  libsim.Init(); // does not seem to be called automatically, what's wrong?
  PrintInfo();
  Run();
}

func Run(){
 const(N0 = 64; 
       N1 = 64;
       N2 = 4;
  );

  size := []int{N0, N1, N2};
  size3D := []int{3, N0, N1, N2};
  s := NewFieldPlan(size, []float{5E-9, 5E-9, 5E-9}, 800E3, 1.3E-11);
  s.PrintInfo();
 
  M := NewTensor4(size3D);
  euler := NewEuler(M, s, 0.01);
  
  SetAll(M.Component(Y), 1.);
  for i:=0; i<len(M.Component(X).List())/2; i++{
    M.Component(Y).List()[i] = -1.
  }
  M.Component(Z).List()[64*32+16] = -1.;
  //SetAll(M.Component(Z), 0.2);
  

  t:=0;
  for i:=0; i<1000; i++{
    Print(t, " ");
    for j:=0; j<100; j++{
      euler.Step();
      t++;
    }
    PrintVectors(FOpen(Sprintf("/home/arne/Desktop/vortex")), M);
  }
  
}


func PrintInfo(){
  Println("Go frontend");
//   Print("libsim build:      ");
//   Println(libsim.Build());
//   Print("       precission: ");
//   Println(libsim.Precission());
}








 
  ////set magnetized sphere
//   center := NewVector();
//   center.Set(float(N0+1)/2., float(N1+1)/2., float(N2+1)/2.);
//   r := NewVector();
//   mx := m.Component(0);
//   for it := NewIterator(mx); it.HasNext(); it.Next(){
//     r.Set(float(it.Index()[0]), float(it.Index()[1]), float(it.Index()[2]));
//     r.Sub(center);
//     if r.Norm() < float(N0)/2. {
//       Set(mx, it.Index(), 1.);
//     }
//   }







