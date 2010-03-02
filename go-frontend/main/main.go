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
 const(N0 = 16; 
       N1 = 16;
       N2 = 1;
  );

  size := []int{N0, N1, N2};
  size3D := []int{3, N0, N1, N2};
  field := NewFieldPlan(size, []float{5E-9, 5E-9, 5E-9}, 800E3, 1.3E-11);
  field.PrintInfo();
 
  M := NewTensor4(size3D);
  euler := NewRKF(M, field, 0.1);
  
  SetAll(M.Component(Y), 1.);
  for i:=0; i<len(M.Component(X).List())/2; i++{
    M.Component(Y).List()[i] = -1.
  }
  //M.Component(Z).List()[0] = -1.;
  //SetAll(M.Component(Z), 0.2);
  

  t:=0;
  for i:=0; i<10; i++{
    Print(t, " ");
    PrintVectors(FOpen(Sprintf("/home/arne/Desktop/vortex")), M);
    for j:=0; j<50; j++{
      euler.Step();
      t++;
    }
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







