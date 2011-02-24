//  Copyright 2010  Ben Van de Wiele
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
//   "fmt"
)


type SemiAnal1 struct {
	*Sim
	m2 *DevTensor
	h2 *DevTensor
	Reductor
}

func NewSemiAnal1(sim *Sim) *SemiAnal1 {
	this := new(SemiAnal1)
	this.Sim = sim
	this.m2 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
	// 	this.h2 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
	//  this.Reductor.InitMaxVector(sim.Backend, sim.size[X]*sim.size[Y]*sim.size[Z])
	this.Reductor.InitMaxAbs(sim.Backend, prod(sim.size4D[0:]))
	return this
}

func (s *SemiAnal1) Step() {
	m1 := s.mDev
	m2 := s.m2
	h := s.hDev
	// 	h2 := s.h2
	//   fmt.Println("Step", s.dt, "oiuuigy" )
	if s.steps == 0 {
		s.calcHeff(m1, h)
		s.SemianalStep(m1, m2, h, s.dt/2.0, s.alpha)
		s.calcHeff(m2, h)
		s.SemianalStep(m1, m1, h, s.dt, s.alpha)
	/*	s.Torque(m2, h)
		s.torque = s.Reduce(h)*/
	} else {

		s.calcHeff(m1, h)
		s.SemianalStep(m2, m2, h, s.dt, s.alpha)
		s.calcHeff(m2, h)
		s.SemianalStep(m1, m1, h, s.dt, s.alpha)
// 		s.Torque(m2, h)
// 		s.torque = s.Reduce(h)

	}

	if s.steps%100 == 0 {
		s.Normalize(m1)
		s.Normalize(m2)
	}
}


func (this *SemiAnal1) String() string {
	return "Semianlytical 1"
}


// predictor corrector *************************************************************
type SemiAnal2 struct {
	*Sim
	m2 *DevTensor
	//   h2 *DevTensor
	m3 *DevTensor
	//   h3 *DevTensor
	Reductor
}

func NewSemiAnal2(sim *Sim) *SemiAnal2 {
	this := new(SemiAnal2)
	this.Sim = sim
	this.m2 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
	//   this.h2 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
	this.m3 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
	//   this.h3 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
	//  this.Reductor.InitMaxVector(sim.Backend, sim.size[X]*sim.size[Y]*sim.size[Z])
	this.Reductor.InitMaxAbs(sim.Backend, prod(sim.size4D[0:]))
	return this
}

func (s *SemiAnal2) Step() {
	m1 := s.mDev
	h1 := s.hDev
	m2 := s.m2
	//   h2 := s.h2
	m3 := s.m3
	//   h3 := s.h3

	if s.steps == 0 {
		s.calcHeff(m1, h1)
		s.SemianalStep(m1, m2, h1, s.dt/2.0, s.alpha)
		s.calcHeff(m2, s.hDev)
		s.SemianalStep(m1, m1, h1, s.dt, s.alpha)
	} else {

		s.calcHeff(m1, h1)
		s.SemianalStep(m2, m3, h1, s.dt, s.alpha)
		s.LinearCombination(m3, m2, 0.5, 0.5)
		s.calcHeff(m3, h1)
		s.SemianalStep(m2, m2, h1, s.dt, s.alpha)

		s.calcHeff(m2, h1)
		s.SemianalStep(m1, m3, h1, s.dt, s.alpha)
		s.LinearCombination(m3, m1, 0.5, 0.5)
		s.calcHeff(m3, h1)
		s.SemianalStep(m1, m1, h1, s.dt, s.alpha)

		//     s.calcHeff(m1, h1)
		//     s.SemianalStep(m2, m3, h1, s.dt, s.alpha)
		//     s.calcHeff(m2, h2)
		//     s.calcHeff(m3, h3)
		//     s.LinearCombination(h3, h2, 0.5, 0.5)
		//     s.SemianalStep(m2, m2, h3, s.dt, s.alpha)
		// 
		//     s.calcHeff(m2, h2)
		//     s.SemianalStep(m1, m3, h2, s.dt, s.alpha)
		//     s.calcHeff(m1, h1)
		//     s.calcHeff(m3, h3)
		//     s.LinearCombination(h3, h1, 0.5, 0.5)
		//     s.SemianalStep(m1, m1, h3, s.dt, s.alpha)


	}

	//   s.calcHeff(m1, s.hDev)
	//   s.SemianalStep(m1, m2, s.hDev, s.dt, s.alpha)
	//   s.calcHeff(m2, h2)
	//   s.LinearCombination(s.hDev, h2, 0.5, 0.5)
	//   s.SemianalStep(m1, m1, s.hDev, s.dt, s.alpha)


	if s.steps%100 == 0 {
		s.Normalize(m1)
		s.Normalize(m2)
	}
}
//*****************************************************************************


// anal FW ********************************************************************
// type SemiAnal2 struct {
//   *Sim
// }
// 
// func NewSemiAnal2(sim *Sim) *SemiAnal2 {
//   this := new(SemiAnal2)
//   this.Sim = sim
//   return this
// }
// 
// func (s *SemiAnal2) Step() {
//   m := s.mDev
//   h := s.hDev
//   s.calcHeff(m, h)
//   s.SemianalStep(m, m, h, s.dt, s.alpha, m.length/3)
// 
//   
//   if (s.steps%100 == 0){
//     s.Normalize(m)
//   }
//   
// }

//*****************************************************************************


/*
type SemiAnal2 struct {
  *Sim
  m2 *DevTensor
  m3 *DevTensor
  h2 *DevTensor
  h3 *DevTensor
}
25
func NewSemiAnal2(sim *Sim) *SemiAnal2 {
  this := new(SemiAnal2)
  this.Sim = sim
  this.m2 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
  this.m3 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
  this.h2 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
  this.h3 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
  return this
}

func (s *SemiAnal2) Step() {
  m1 := s.mDev
  m2 := s.m2
  m3 := s.m3
  h1 := s.hDev
  h2 := s.h2
  h3 := s.h3

  if s.steps == 0{
      //set up m2
    s.calcHeff(m1, h1)
    s.SemianalStep(m1, m2, h1, s.dt/2.0, s.alpha)
      //predictor m1
    s.calcHeff(m2, h2)
    s.SemianalStep(m1, m3, h2, s.dt, s.alpha)
      //corrector m1
    s.calcHeff(m3, h3)
    s.LinearCombination(h1, h3, 0.5, 0.5)    // h1 = 0.25*h1 + 0.5*h2 + 0.25*h3
    s.LinearCombination(h1, h2, 0.5, 0.5)
    s.SemianalStep(m1, m1, h1, s.dt, s.alpha)

  } else{

      //predictor m2
    s.calcHeff(m1, h1)
    s.SemianalStep(m2, m3, h1, s.dt, s.alpha)
      //corrector m2
    s.calcHeff(m3, h3)
    s.LinearCombination(h2, h3, 0.5, 0.5)    // h2 = 0.25*h2 + 0.5*h1 + 0.25*h3
    s.LinearCombination(h2, h1, 0.5, 0.5)
    s.SemianalStep(m2, m2, h2, s.dt, s.alpha)

      //predictor m1
    s.calcHeff(m2, h2)
    s.SemianalStep(m1, m3, h2, s.dt, s.alpha)
      //corrector m1
    s.calcHeff(m3, h3)
    s.LinearCombination(h1, h3, 0.5, 0.5)    // h1 = 0.25*h1 + 0.5*h2 + 0.25*h3
    s.LinearCombination(h1, h2, 0.5, 0.5)
    s.SemianalStep(m1, m1, h1, s.dt, s.alpha)

  }

  if (s.steps%100 == 0){
    s.Normalize(m1)
    s.Normalize(m2)
  }

}*/


// type SemiAnal2 struct {
//   *Sim
//   m2 *DevTensor
//   m_pred *DevTensor
//   h2 *DevTensor
// }
// 
// func NewSemiAnal2(sim *Sim) *SemiAnal2 {
//   this := new(SemiAnal2)
//   this.Sim = sim
//   this.m2 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
//   this.m_pred = NewTensor(sim.Backend, Size4D(sim.size[0:]))
//   this.h2 = NewTensor(sim.Backend, Size4D(sim.size[0:]))
//   return this
// }
// 
// func (s *SemiAnal2) Step() {
//   m1 := s.mDev
//   m2 := s.m2
//   m_pred := s.m_pred
//   h1 := s.hDev
//   h2 := s.h2
//   
//   if s.steps == 0{
//       //set up m2
//     s.calcHeff(m1, h1)
//     s.SemianalStep(m1, m2, h1, s.dt/2.0, s.alpha)
//       //predictor m1
//     s.calcHeff(m2, h2)
//     s.SemianalStep(m1, m_pred, h2, s.dt, s.alpha)
//       //corrector m1
//     s.calcHeff(m_pred, h2)
//     s.LinearCombination(h2, h1, 0.5, 0.5)
//     s.SemianalStep(m1, m1, h2, s.dt, s.alpha)
// 
//   } else{
//     
//       //predictor m2
//     s.calcHeff(m1, h1)
//     s.SemianalStep(m2, m_pred, h1, s.dt, s.alpha)
//       //corrector m2
//     s.calcHeff(m_pred, h1)
//     s.LinearCombination(h1, h2, 0.5, 0.5)
//     s.SemianalStep(m2, m2, h1, s.dt, s.alpha)
// 
//       //predictor m1
//     s.calcHeff(m2, h2)
//     s.SemianalStep(m1, m_pred, h2, s.dt, s.alpha)
//       //corrector m1
//     s.calcHeff(m_pred, h2)
//     s.LinearCombination(h2, h1, 0.5, 0.5)
//     s.SemianalStep(m1, m1, h2, s.dt, s.alpha)
// 
//   }
//   
//   if (s.steps%100 == 0){
//     s.Normalize(m1)
//     s.Normalize(m2)
//   }
//   
// }



func (this *SemiAnal2) String() string {
	return "Semianlytical 2"
}
