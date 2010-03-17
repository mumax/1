package fft

import( . "../assert";
	. "../tensor";
	"../libsim";
	"unsafe";
	//. "fmt";
);

/*
 * An FTPlan wraps a 3D FFTW plan in a safe way. It's OK for the data to have a lower dimension.
 */


// TODO: refactor to fft.Plan
type FTPlan struct{
  real, transformed *Tensor3;			// original (real) and transformed data	
  plan unsafe.Pointer;				// FFTW plan for transforming
}


/** Plan for forward-transfoming real to transformed. Transformed should be larger than real. */

func NewForwardPlan(real, transformed *Tensor3) *FTPlan{
  checkSizes(real, transformed);
  size := real.Size();
  plan := libsim.FFTInitForward(size[0], size[1], size[2], DataAddress(real), DataAddress(transformed));
  Zero(real);
  Zero(transformed);
  return &FTPlan{real, transformed, plan};
}


/** Plan for backward-transfoming transformed to real.  Transformed should be larger than real.*/

func NewBackwardPlan(real, transformed *Tensor3) *FTPlan{
  checkSizes(real, transformed);
  size := real.Size();
  plan := libsim.FFTInitBackward(size[0], size[1], size[2], DataAddress(transformed), DataAddress(real));
  Zero(real);
  Zero(transformed);
  return &FTPlan{real, transformed, plan};
}


/** Plan for forward-transfoming in-place. */
// Todo: CHECK SIZES!
func NewInplaceForwardPlan(data *Tensor3) *FTPlan{
  //checkSizes(real, transformed);
  size := data.Size();
  // size is supposed to be already padded with one extra complex number in last dimension
  // logical size is thus smaller:
  plan := libsim.FFTInitForward(size[0], size[1], size[2]-2, DataAddress(data), DataAddress(data));
  Zero(data);
  return &FTPlan{data, data, plan};
}


/** Plan for backward-transfoming in-place. */
// Todo: CHECK SIZES!
func NewInplaceBackwardPlan(data *Tensor3) *FTPlan{
  //checkSizes(real, transformed);
  size := data.Size();
  // size is supposed to be already padded with one extra complex number in last dimension
  // logical size is thus smaller:
  plan := libsim.FFTInitBackward(size[0], size[1], size[2]-2, DataAddress(data), DataAddress(data));
  Zero(data);
  return &FTPlan{data, data, plan};
}


/** Execute transform */

func (p *FTPlan) Execute(){
  //Println(unsafe.Pointer(DataAddress(p.real)), "<-FFT->", //unsafe.Pointer(DataAddress(p.transformed)));
  libsim.FFTExecute(p.plan);
}


/////////////////////////////////// private subroutines

/* Make sure transformed size is 1 complex number larger. */
func checkSizes(real, transformed *Tensor3){
    r := real.Size();
    t := transformed.Size();
    AssertMsg(!(r[0] != t[0] || r[1] != t[1] || r[2]+2 != t[2]), "NewPlan: Size of transformed block should be 1 complex number larger than real data");
}


