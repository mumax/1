package sim

/* 
 * This file contains numerous functions that return micromagnetic kernels.
 *
 * A kernel is a rank 5 Tensor: K[S][D][x][y][z].
 * S and D are source and destination directions, ranging from 0 (X) to 2 (Z).
 * K[S][D][x][y][z] is the D-the component of the magnetic field at position
 * (x,y,z) due to a unit spin along direction S, at the origin.
 *
 * The kernel is twice the size of the magnetization field we want to convolve it with.
 * The indices are wrapped: a negative index i is stored at N-abs(i), with N
 * the total size in that direction.
 *
 * Idea: we migth calculate in the kernel in double precession and only round it
 * just before it is returned, or even after doing the FFT. Because it is used over
 * and over, this small gain in accuracy *might* be worth it.
 */

import(
  . "../tensor";
  . "../fft";
  . "math";
)

/** 
 * With this kernel, cells are treated as homogenously magnetized cuboids.
 * The magnetic charges on the faces are integrated by just taking one
 * point at the center of the face. It is thus not a very accurate kernel,
 * but OK for debugging.
 */
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


/**
 * With this kernel, each "cell" contains a magnetic point-dipole.
 * This is more useful for spin-lattice simulations than micromagnetics.
 * The self-contribution is (anti)parallel to m and thus does not
 * influence the dynamics.
 */
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

/* --------- Internal functions --------- */

func wrap(number, max int) int{
  for number < 0{
    number += max;
  }
  return number;
}


func FSqrt(x float64) float{
  return float(Sqrt(x));
}
