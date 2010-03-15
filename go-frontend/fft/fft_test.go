package fft

import( . "testing";
	. "../tensor";
	"rand";
	. "math";
	"fmt";
	//. "os";
)

/*
 * FFT Unit Tests
 */


func TestFFTPlan(t *T){
  const(N0 = 3; 
        N1 = 4;
	N2 = 5
  );

  m:=NewTensor3([]int{N0, N1, N2});
  h:=NewTensor3([]int{N0, N1, N2+2});
  fw := NewForwardPlan(m, h);
  bw := NewBackwardPlan(m, h);


  ma := m.List();
  for i:= range(m.List()){
    ma[i] = rand.Float()+0.001; // so it's not zero
   }
  //m.Array()[1][2][3] = 123;

  //PrintTensor(Stdout, m);
  orig := Copy(m);
  //PrintTensor(Stdout, m);

  //orig.Print(Stdout);
  fw.Execute();

  //PrintTensor(Stdout, h);

  bw.Execute();

  //scale
  for i:= range(ma){
    ma[i] /= float(N0*N1*N2);
  }

  //PrintTensor(Stdout, m);

  var rms float64 = 0.;
  for i:= range(m.List()){
    // we test if the tranformed + backtransformed data equals the original
    // must take into account scale factor
    // also, we do not allow zero's, as it is too dangerous to be comparing to zero arrays by accident.
    //if Fabs(float64(ma[i] - orig.List()[i])) > 0.0001  || m.List()[i] == 0.{ t.Fail(); }
    rms += float64( (float64(ma[i] - orig.List()[i])) * (float64(ma[i] - orig.List()[i])) );
  }
  rms = Sqrt(rms);
  fmt.Println("FFT rms error: ", rms);

  //m.Real().Print(Stdout);*/

}
