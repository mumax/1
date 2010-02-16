package fft

import(
	. "../assert";
	. "../tensor";
	//. "fmt";
	//. "os"
);

/*********************************************
 * Convolution plan
 *********************************************/


type ConvPlan struct{

    /** The input data and the convolved output data, tentatively called m for magnetization and h for field. They are to be fields of 3-dimensional vectors. */
    //m, h *Tensor4;

    size []int;

   /** Transformed magnetization component m_i, re-used for m_0, m_1, m_2 (allocating three separate buffers would be a waste of space). This space is zero-padded to be twice the size of the input data in all directions. Plus, the last dimension is even 2 elements larger to allow an in-place FFT. */
   ft_m_i *Tensor3;

   /** Forward plan for the zero-padded magnetization components. Inplace in ft_m_i. */
   forward *FTPlan;

   /** Transformed Kernel */
   ft_kernel [3][3]*Tensor3; // todo: make tensor5

   /** Transformed total demag field due to all the magnetization components i: h[j] = sum_i h_i[j] */
   ft_h [3]*Tensor3; // make tensor4
  
   /** demag field of one component, re-used for all components, padded with spoiled data. Non-spoiled region of interest gets summed up in h */
   //h_i_padded *Block;

   /** Backward plans for ft_h[i]. Sources: ft_h[i], destinations: h[i]. */
   backward [3]*FTPlan;

}


/** New convolution plan for a vector convolution of source data with a tensor kernel, result stored in field. Kernel is not required to be of type Tensor5 but can be a general tensor so it may be calculated on-the-fly. */

func NewConvPlan(size []int, kernel Tensor) *ConvPlan{
  AssertMsg(Rank(kernel) == 5, "Need kernel of rank 5");
	
  //size := source.Component(0).Size();
  //AssertMsg(EqualSize(source.Size(), field.Size()), "NewConvPlan: fields need to be equal size");
  plan := new(ConvPlan);
  

  plan.size = size;
  // source and destination field
//   plan.m = source;
//   plan.h = field;

  // stores only one magnetization component at a time, padded with zero's
  paddedsize := PadSize(size);//scale(2, plan.m.Component(0).Size());

  // stores one magnetization component, transformed
  paddedcomplexsize := []int{paddedsize[0], paddedsize[1], paddedsize[2]+2};
  plan.ft_m_i = NewTensor3(paddedcomplexsize);

  // transforms padded magnetization component to transformed magnetization component, one at a time
  plan.forward = NewInplaceForwardPlan(plan.ft_m_i);	// in-place to save memory
							// perhaps we should keep the option open to have it out-of-place, if this gives better performance at the cost of memory.
  plan.ft_h = [3]*Tensor3{};
  for i:=range(plan.ft_h){
    plan.ft_h[i] = NewTensor3(paddedcomplexsize);
  }

  plan.backward = [3]*FTPlan{};
  for i:=range(plan.backward){
    plan.backward[i] = NewInplaceBackwardPlan(plan.ft_h[i]);
  }

  // then make fft plans (first), fill data (second), execute plan (third)
  for s:=0; s<3; s++{
    for d:=0; d<3; d++{
      plan.ft_kernel[s][d] = NewTensor3(paddedcomplexsize);
      kernplan := NewInplaceForwardPlan(plan.ft_kernel[s][d]);
      for i:=0; i<paddedsize[0]; i++{
	for j:=0; j<paddedsize[1]; j++{
	  for k:=0; k<paddedsize[2]; k++{
	    Set(plan.ft_kernel[s][d], []int{i,j,k}, Get(kernel, s, d, i, j, k) / (float(paddedsize[0]*paddedsize[1]*paddedsize[2]))); // normalization to be spread over forward+backward?
	  }
	}
      }
      //Println("K", s, d);
      //PrintTensor(Stdout, &plan.ft_kernel[s][d].StoredTensor);
      kernplan.Execute(); // in-place transform of kernel component.
      //Println("K", s, d);
      //PrintTensor(Stdout, &plan.ft_kernel[s][d].StoredTensor);
    }
  }



  return plan;
}


/** Total number of logical elements, used for, e.g., normalization. */

func (p *ConvPlan) N() int{
  return p.size[0] * p.size[1] * p.size[2];
}


/** Execute the convolution. */

func (p *ConvPlan) Execute(m, h *Tensor4){
  
  // Zero-out field components
  for i:=0; i<3; i++{
    Zero(p.ft_h[i]);
  }

  for i:=0; i<3; i++{
    // zero-out the padded magnetization buffer first
    Zero(p.ft_m_i);

    // then copy the current magnetization component in the padded magnetization buffer
   CopyInto(m.Component(i).Array(), p.ft_m_i.Array());

    // in-place FFT of the padded magnetization
    p.forward.Execute();
   
    // apply kernel multiplication to FFT'ed magnetization and add to FFT'ed H-components
    ft_m_i := p.ft_m_i.List();
    for j:=0; j<3; j++{
      ft_h_j := p.ft_h[j].List();
      for e:=0; e<len(ft_m_i); e+=2{
	rea := ft_m_i[e];
	reb := p.ft_kernel[i][j].List()[e];
	ima := ft_m_i[e+1];
	imb := p.ft_kernel[i][j].List()[e+1];
	ft_h_j[e] +=  rea*reb - ima*imb;
	ft_h_j[e+1] +=  rea*imb + ima*reb;
      }
    }
  }

  
  for i:=0; i<3; i++{
    // Inplace backtransform of each of the padded H-buffers
    p.backward[i].Execute();

    // Copy region of interest (non-padding space) to destination
    CopyFrom(p.ft_h[i].Array(), h.Component(i).Array());
  }
}


func scale(r int, array []int) []int{
  scaled := make([]int, len(array));
  for i:= range(scaled){
    scaled[i] = r*array[i];
  }
  return scaled;
}

/** Padded size is, strictly speaking 2*N-1 in each direction. We return 2*N however, because 2*N-1 may be prime. We may change this to return 1 for the case of N=1, however. */

func PadSize(size []int) []int{
  paddedsize := make([]int, len(size));
  for i:= range(size){
  // THIS SEEMS TO BREAK THINGS:
//     if(size[i] < 4){
//       paddedsize[i] = 2*size[i]-1; // sizes 1, 2, 3, 4, 5 should be OK
//     }
//      else{
      paddedsize[i] = 2*size[i];
     //}
   }
  return paddedsize;
}


/*
 * Private
 */

func CopyInto(source, dest [][][]float){
  Assert(len(dest) >= len(source)); // & ...

    for i:= range(source){
      for j:= range(source[i]){
	for k:= range(source[i][j]){
	  dest[i][j][k] = source[i][j][k];
	}
      }
    }
}


/** Copy from a block with equal or larger Size. Unused elements in the larger block are ignored */

func CopyFrom(source, dest [][][]float){
  Assert(len(dest) <= len(source)); // & ...

    for i:= range(dest){
      for j:= range(dest[i]){
	for k:= range(dest[i][j]){
	  dest[i][j][k] = source[i][j][k];
	}
      }
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
