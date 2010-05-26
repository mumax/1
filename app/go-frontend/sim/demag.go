package sim

import(
  . "../tensor";
  . "math";
)



/** 
TODO: uses only one point (yet), should be made more accurate!
Magnetostatic field at position r (integer, number of cellsizes away form source) for a given source magnetization direction m (X, Y, or Z) */
func faceIntegral(B, R *Vector, cellsize[] float, s int){
  n := 8;					// number of integration points = n^2
  u, v, w := s, (s+1)%3, (s+2)%3;		// u = direction of source (s), v & w are the orthogonal directions
  R2 := NewVector();
  pole := NewVector();				// position of point charge on the surface
  

  surface := cellsize[v] * cellsize[w]; 	// the two directions perpendicular to direction s
  charge := surface;

  pu1 := cellsize[u] / 2.;			// positive pole
  pu2 := -pu1;					// negative pole

  B.Set(0., 0., 0.);				// accumulates magnetic field
  for i:=0; i<n; i++{
    pv := -(cellsize[v]/2.) + cellsize[v]/float(2*n) + float(i)*(cellsize[v]/float(n));
    for j:=0; j<n; j++{
      pw := -(cellsize[w]/2.) + cellsize[w]/float(2*n) + float(j)*(cellsize[w]/float(n));
      
      pole.Component[u] = pu1;
      pole.Component[v] = pv;
      pole.Component[w] = pw;
  
      R2.SetTo(R);
      R2.Sub(pole);
      r := R2.Norm();
      R2.Normalize();
      R2.Scale(charge / (4*Pi*r*r));
      B.Add(R2); 

      pole.Component[u] = pu2;
  
      R2.SetTo(R);
      R2.Sub(pole);
      r = R2.Norm();
      R2.Normalize();
      R2.Scale(-charge / (4*Pi*r*r));
      B.Add(R2); 
    }
  }
  B.Scale(1./(float(n*n))); // n^2 integration points
}



/** Integrates the demag field based on multiple points per face. */
func FaceKernel(unpaddedsize []int, cellsize []float) *Tensor5{
  size := PadSize(unpaddedsize);
  k := NewTensor5([]int{3, 3, size[0], size[1], size[2]});
  B := NewVector();
  R := NewVector();
  
  for s:=0; s<3; s++{					// source index Ksdxyz
    for x:=-(size[X]-1)/2; x<=size[X]/2; x++{		 // in each dimension, go from -(size-1)/2 to size/2, wrapped. 
      xw := wrap(x, size[X]);
      for y:=-(size[Y]-1)/2; y<=size[Y]/2; y++{
	yw := wrap(y, size[Y]);
	for z:=-(size[Z]-1)/2; z<=size[Z]/2; z++{
	  zw := wrap(z, size[Z]);
	  R.Set(float(x) * cellsize[X], float(y) * cellsize[Y], float(z) * cellsize[Z]);

	  faceIntegral(B, R, cellsize, s);

	  for d:=0; d<3; d++{				// destination index Ksdxyz
	    k.Array()[s][d][xw][yw][zw] = B.Component[d]; 
	  }

	}
      }
    }
  }

  return k;
}


/** DEBUG: 'integrates' the magnetostatic field out of just one point per face. */
// func PointKernel(unpaddedsize []int, cellsize []float) *Tensor5{
//   size := PadSize(unpaddedsize);
//   k := NewTensor5([]int{3, 3, size[0], size[1], size[2]});
// 
//   B := NewVector();
//   R := NewVector();
//   
//   pole := make([]*Vector, 2);
//   
//   for s:=0; s<3; s++{	// source index Ksdxyz
// 
//     surface := cellsize[(s+1)%3] * cellsize[(s+2)%3]; 	// the two directions perpendicular to direction s
//     charge := surface;					// unit charge density on surface
//     pole[0] = UnitVector(s);
//     pole[0].Scale(cellsize[s]/2.);
//     pole[1] = UnitVector(s);
//     pole[1].Scale(-cellsize[s]/2.);
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
// 	    R.Set(float(x) * cellsize[X], float(y) * cellsize[Y], float(z) * cellsize[Z]);
// 	    R.Sub(pole[p]);
// 	    r := R.Norm();
// 	    R.Normalize();
// 	    R.Scale(charge / (4*Pi*r*r));
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
// func DipoleKernel(unpaddedsize []int) *Tensor5{
//   size := PadSize(unpaddedsize);
//   k := NewTensor5([]int{3, 3, size[0], size[1], size[2]});
//     
//   R := NewVector();
//   B := NewVector();
// 
//   for s:=0; s<3; s++{	// source index Ksdxyz
//   m:=UnitVector(s);	// unit vector along the source axis, e.g.: x.
//   
//   // in each dimension, go from -(size-1)/2 to size/2, wrapped. 
//   for x:=-(size[X]-1)/2; x<=size[X]/2; x++{
//     xw := wrap(x, size[X]);
//     for y:=-(size[Y]-1)/2; y<=size[Y]/2; y++{
//       yw := wrap(y, size[Y]);
//       for z:=-(size[Z]-1)/2; z<=size[Z]/2; z++{
// 	zw := wrap(z, size[Z]);
// 	if !(xw==0 && yw==0 && zw==0){ // exclude self-contribution
// 	  // B = \mu_0 / (4 \pi r^3) * (3(m \dot \hat r)\hat r -m)
// 	  //B = \frac {\mu_0} {4\pi r^3}  ( 3 ( m \dot \hat{r}) \hat{r} - m)
// 	  R.Set(float(x), float(y), float(z));
// 	  r := R.Norm();
// 	  R.Normalize();
// 	  B.SetTo(R);
// 	  B.Scale(3.* m.Dot(R));
// 	  B.Sub(m);
// 	  B.Scale(1./(4.*Pi*r*r*r));
// 	  for d:=0; d<3; d++{	// destination index Ksdxyz
// 	    k.Array()[s][d][xw][yw][zw] = B.Component[d]; 
// 	  }
// 	}
//       }
//     }
// 
//   }
//   }
// 
//   // self-contributions (does not matter as it is parallel to m)
//   for i:=0; i<3; i++{
//     k.Array()[i][i][0][0][0] = -1. 
//   }
// 
//   return k;
// }
