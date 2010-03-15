package main
  
import( 
       . "../tensor";
       //. "../fft";
       . "../sim";
       //"../libsim";
       "fmt";
       . "os";
       //. "../util";
)

func main() {
  //libsim.Init(); // does not seem to be called automatically, what's wrong > init(), not Init() !
  PrintInfo();
  Run();
}

func Run(){
 const(N0 = 32; 
       N1 = 32;
       N2 = 4;
  );

  size := []int{N0, N1, N2};
  size3D := []int{3, N0, N1, N2};
  
  M := NewTensor4(size3D);
  
  field := NewFieldPlan(size, []float{5E-9, 5E-9, 5E-9}, 800E3, 1.3E-11);
  field.PrintInfo();
 
  
  euler := NewRK4MaxDm(M, field, 0.02);
  
  SetAll(M.Component(Y), 1.);
  for i:=0; i<len(M.Component(X).List())/2; i++{
    M.Component(Y).List()[i] = -1.
  }
  
  Write(FOpen("/home/arne/Desktop/vortex"), M);
  Print(Stdout, M);
  
  M.Component(Z).List()[0] = -1.;
  SetAll(M.Component(Z), 0.2);
  

  t:=0;
  for i:=0; i<50; i++{
    fmt.Print(t, " ");
    Write(FOpen("/home/arne/Desktop/vortex"), M);
    for j:=0; j<100; j++{
      euler.Step();
      t++;
    }
  }
}


func PrintInfo(){
  fmt.Println("Go frontend");
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







