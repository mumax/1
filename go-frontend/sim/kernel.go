package sim

import(
  . "../tensor";
  . "../fft";
  . "math";
  //. "fmt";
)


func PointKernel(unpaddedsize []int, cubesize float) *Tensor5{
  size := PadSize(unpaddedsize);
  k := NewTensor5([]int{3, 3, size[0], size[1], size[2]});
  

  B := NewVector();
  R := NewVector();
  
  pole := make([]*Vector, 2);
  
  for s:=0; s<3; s++{	// source index Ksdxyz

    surface := cubesize * cubesize;
    charge := surface;
    pole[0] = UnitVector(s);
    pole[0].Scale(cubesize/2.);
    pole[1] = UnitVector(s);
    pole[1].Scale(-cubesize/2.);

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

  // self-contributions
  for i:=0; i<3; i++{
    k.Array()[i][i][0][0][0] = -1. 
  }

  return k;
}


/*
func UnwrappedDipoleKernel(unpaddedsize []int) *Tensor5{
  size := PadSize(unpaddedsize);
  k := NewTensor5([]int{3, 3, size[0], size[1], size[2]});
    
  R := NewVector();
  B := NewVector();

  for s:=0; s<3; s++{	// source index Ksdxyz
  m:=UnitVector(s);	// unit vector along the source axis, e.g.: x.
  
  // in each dimension, go from -(size-1)/2 to size/2, wrapped. 
  for x:=0; x<size[X]; x++{
    for y:=0; y<size[Y]; y++{
      for z:=0; z<size[Z]; z++{
	if !(x==0 && y==0 && z==0){ // exclude self-contribution
	  // B = \mu_0 / (4 \pi r^3) * (3(m \dot \hat r)\hat r -m)
	  R.Set(float(x), float(y), float(z));
	  r := R.Norm();
	  R.Normalize();
	  B.SetTo(R);
	  B.Scale(3.*m.Dot(R));
	  B.Sub(m);
	  B.Scale(1./(4.*Pi*r*r*r));
	  for d:=0; d<3; d++{	// destination index Ksdxyz
	    k.Array()[s][d][x][y][z] = B.Component[d]; 
	  }
	}
      }
    }

  }

  }

  return k;
}*/


func wrap(number, max int) int{
  for number < 0{
    number += max;
  }
  return number;
}


func FSqrt(x float64) float{
  return float(Sqrt(x));
}

/** wraps negative indices to padded positive indices */

// func wrap(paddedindex, paddedsize []int) []int{
//   wrapped := make([]int, len(paddedsize));
//   for i:=range(paddedsize){
//     for paddedindex[i] < 0{
//       paddedindex[i] += paddedsize[i];
//     }
//   }
//   return wrapped;
// }
