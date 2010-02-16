package sim

import(
  . "../tensor";
  . "../fft";
  . "math";
  . "fmt";
)



/** Magnetostatic field at position r (integer, number of cellsizes away form source) for a given source magnetization direction m (X, Y, or Z) */
func faceIntegral(cellsize[] float, r []int, mdir int) [3]float{

  pole := make([]*Vector, 2);
  B := NewVector();
  R := NewVector();

  surface := cellsize[(mdir+1)%3] * cellsize[(mdir+2)%3]; 	// the two directions perpendicular to direction s
  charge := surface;					// unit charge density on surface
  pole[0] = UnitVector(mdir);
  pole[0].Scale(cellsize[mdir]/2.);
  pole[1] = UnitVector(mdir);
  pole[1].Scale(-cellsize[mdir]/2.);

  B.Set(0., 0., 0.);
  for p:=0; p<2; p++{
    R.Set(float(r[X]), float(r[Y]), float(r[Z]));
    R.Sub(pole[p]);
    r := R.Norm();
    R.Normalize();
    R.Divide(4*charge*Pi*r*r);
    if p == 1 {R.Scale(-1.)};
    B.Add(R); 
  }
  return [3]float{B.Component[0], B.Component[1], B.Component[2]};
}



/** DEBUG: integration with just one point */
// func pointIntegral(cellsize[] float, r []int, mdir int) [3]float{
// 
//   pole := make([]*Vector, 2);
//   B := NewVector();
//   R := NewVector();
// 
//   surface := cellsize[(mdir+1)%3] * cellsize[(mdir+2)%3]; 	// the two directions perpendicular to direction s
//   charge := surface;					// unit charge density on surface
//   pole[0] = UnitVector(mdir);
//   pole[0].Scale(cellsize[mdir]/2.);
//   pole[1] = UnitVector(mdir);
//   pole[1].Scale(-cellsize[mdir]/2.);
// 
//   B.Set(0., 0., 0.);
//   for p:=0; p<2; p++{
//     R.Set(float(r[X]), float(r[Y]), float(r[Z]));
//     R.Sub(pole[p]);
//     r := R.Norm();
//     R.Normalize();
//     R.Divide(4*charge*Pi*r*r);
//     if p == 1 {R.Scale(-1.)};
//     B.Add(R); 
//   }
//   return [3]float{B.Component[0], B.Component[1], B.Component[2]};
// }

/** Integrates the demag field based on multiple points per face. */
func FaceKernel(unpaddedsize []int, cellsize []float) *Tensor5{
  Println("FaceKernel(", unpaddedsize, cellsize, ")");
  size := PadSize(unpaddedsize);
  k := NewTensor5([]int{3, 3, size[0], size[1], size[2]});
  
  for s:=0; s<3; s++{	// source index Ksdxyz
    // in each dimension, go from -(size-1)/2 to size/2, wrapped. 
    for x:=-(size[X]-1)/2; x<=size[X]/2; x++{
      xw := wrap(x, size[X]);
      for y:=-(size[Y]-1)/2; y<=size[Y]/2; y++{
	yw := wrap(y, size[Y]);
	for z:=-(size[Z]-1)/2; z<=size[Z]/2; z++{
	  zw := wrap(z, size[Z]);
	  B := faceIntegral(cellsize, []int{x, y, z}, s);
	  for d:=0; d<3; d++{	// destination index Ksdxyz
	    k.Array()[s][d][xw][yw][zw] = B[d]; 
	  }
	}
      }
    }
  }

  return k;
}


/** DEBUG: 'integrates' the magnetostatic field out of just one point per face. */
func PointKernel(unpaddedsize []int, cellsize []float) *Tensor5{
  Println("PointKernel(", unpaddedsize, cellsize, ")");
  size := PadSize(unpaddedsize);
  k := NewTensor5([]int{3, 3, size[0], size[1], size[2]});

  B := NewVector();
  R := NewVector();
  
  pole := make([]*Vector, 2);
  
  for s:=0; s<3; s++{	// source index Ksdxyz

    surface := cellsize[(s+1)%3] * cellsize[(s+2)%3]; 	// the two directions perpendicular to direction s
    charge := surface;					// unit charge density on surface
    pole[0] = UnitVector(s);
    pole[0].Scale(cellsize[s]/2.);
    pole[1] = UnitVector(s);
    pole[1].Scale(-cellsize[s]/2.);

    // in each dimension, go from -(size-1)/2 to size/2, wrapped. 
    for x:=-(size[X]-1)/2; x<=size[X]/2; x++{
      xw := wrap(x, size[X]);
      for y:=-(size[Y]-1)/2; y<=size[Y]/2; y++{
	yw := wrap(y, size[Y]);
	for z:=-(size[Z]-1)/2; z<=size[Z]/2; z++{
	  zw := wrap(z, size[Z]);
	  
	  B.Set(0., 0., 0.);
	  for p:=0; p<2; p++{
	    R.Set(float(x), float(y), float(z));
	    R.Sub(pole[p]);
	    r := R.Norm();
	    R.Normalize();
	    R.Divide(4*charge*Pi*r*r);
	    if p == 1 {R.Scale(-1.)};
	    B.Add(R); 
	  }
	  for d:=0; d<3; d++{	// destination index Ksdxyz
	    k.Array()[s][d][xw][yw][zw] = B.Component[d]; 
	  }

	}
      }
    }
  }

  return k;
}

// func PointKernelCubic(unpaddedsize []int, cubesize float) *Tensor5{
//   Println("PointKernelCubic(", unpaddedsize, cubesize, ")");
//   size := PadSize(unpaddedsize);
//   k := NewTensor5([]int{3, 3, size[0], size[1], size[2]});
//   
// 
//   B := NewVector();
//   R := NewVector();
//   
//   pole := make([]*Vector, 2);
//   
//   for s:=0; s<3; s++{	// source index Ksdxyz
// 
//     surface := cubesize * cubesize;
//     charge := surface;
//     pole[0] = UnitVector(s);
//     pole[0].Scale(cubesize/2.);
//     pole[1] = UnitVector(s);
//     pole[1].Scale(-cubesize/2.);
// 
//     // in each dimension, go from -(size-1)/2 to size/2, wrapped. 
//     for x:=-(size[X]-1)/2; x<=size[X]/2; x++{
//       xw := wrap(x, size[X]);
//       for y:=-(size[Y]-1)/2; y<=size[Y]/2; y++{
// 	yw := wrap(y, size[Y]);
// 	for z:=-(size[Z]-1)/2; z<=size[Z]/2; z++{
// 	  zw := wrap(z, size[Z]);
// 	  
// 	  B.Set(0., 0., 0.);
// 	  for p:=0; p<2; p++{
// 	    R.Set(float(x), float(y), float(z));
// 	    R.Sub(pole[p]);
// 	    r := R.Norm();
// 	    R.Normalize();
// 	    R.Divide(4*charge*Pi*r*r);
// 	    if p == 1 {R.Scale(-1.)};
// 	    B.Add(R); 
// 	  }
// 	  for d:=0; d<3; d++{	// destination index Ksdxyz
// 	    k.Array()[s][d][xw][yw][zw] = B.Component[d]; 
// 	  }
// 
// 	}
//       }
//     }
//   }
// 
//   return k;
// }


/** Todo: calculate in double precession and only round in the end? */
func DipoleKernel(unpaddedsize []int) *Tensor5{
  size := PadSize(unpaddedsize);
  k := NewTensor5([]int{3, 3, size[0], size[1], size[2]});
    
  R := NewVector();
  B := NewVector();

  for s:=0; s<3; s++{	// source index Ksdxyz
  m:=UnitVector(s);	// unit vector along the source axis, e.g.: x.
  
  // in each dimension, go from -(size-1)/2 to size/2, wrapped. 
  for x:=-(size[X]-1)/2; x<=size[X]/2; x++{
    xw := wrap(x, size[X]);
    for y:=-(size[Y]-1)/2; y<=size[Y]/2; y++{
      yw := wrap(y, size[Y]);
      for z:=-(size[Z]-1)/2; z<=size[Z]/2; z++{
	zw := wrap(z, size[Z]);
	if !(xw==0 && yw==0 && zw==0){ // exclude self-contribution
	  // B = \mu_0 / (4 \pi r^3) * (3(m \dot \hat r)\hat r -m)
	  //B = \frac {\mu_0} {4\pi r^3}  ( 3 ( m \dot \hat{r}) \hat{r} - m)
	  R.Set(float(x), float(y), float(z));
	  r := R.Norm();
	  R.Normalize();
	  B.SetTo(R);
	  B.Scale(3.* m.Dot(R));
	  B.Sub(m);
	  B.Scale(1./(4.*Pi*r*r*r));
	  for d:=0; d<3; d++{	// destination index Ksdxyz
	    k.Array()[s][d][xw][yw][zw] = B.Component[d]; 
	  }
	}
      }
    }

  }
  }

  // self-contributions (does not matter as it is parallel to m)
  for i:=0; i<3; i++{
    k.Array()[i][i][0][0][0] = -1. 
  }

  return k;
}

func wrap(number, max int) int{
  for number < 0{
    number += max;
  }
  return number;
}

func FSqrt(x float64) float{
  return float(Sqrt(x));
}
