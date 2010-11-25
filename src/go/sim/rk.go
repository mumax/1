//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

import (
"fmt"
)

// General Runge-Kutta solver
type RK struct {
  *Sim
  
	order int

  // y_(n+1) = y_(n) + dt * sum b_i * k_i
  //
  // k_i = torque()
	a  [][]float32
	b  []float32
	c  []float32
	b2 []float32
  k  []*DevTensor
	kdata []uintptr //contiguous array of data pointers from k (kdata[i] = k[i].data)
}


// Euler method
func NewRK1(sim *Sim) *RK{
  rk := newRK(sim, 1)
  rk.b[0] = 1.
  return rk
}



// INTERNAL
func (rk *RK) init(sim *Sim, order int){
  rk.order = order
  rk.Sim = sim
  rk.a = make([][]float32, order)
  for i := range rk.a{
    rk.a[i] = make([]float32, order)
  }
  rk.b = make([]float32, order)
  rk.c = make([]float32, order)
  rk.k = make([]*DevTensor, order)
  rk.kdata = make([]uintptr, order)
  for i:=range rk.k{
    rk.k[i] = NewTensor(sim.Backend, sim.mDev.size)
    rk.kdata[i] = rk.k[i].data
  }
}


// INTERNAL
func (rk *RK) free(){
  for i := range rk.k{
    rk.k[i].Free()
    rk.k[i] = nil
    rk.kdata[i] = 0
  }
}


// INTERNAL
func newRK(sim *Sim, order int) *RK{
  rk := new(RK)
  rk.init(sim, order)
  return rk
}

// INTERNAL
func newAdaptiveRK(sim *Sim, order int) *RK{
  rk :=newRK(sim, order)
  rk.b2 = make([]float32, rk.order)
  return rk
}



func (rk *RK) Step(){
  order := rk.order
  //k := rk.k
  time0 := rk.time
  h := rk.dt
  c := rk.c
  m := rk.mDev
  
  for i:=0; i<order; i++{
    rk.time = time0 + float64(c[i] * h)
    rk.calcHeff(m, rk.h)
    rk.Torque(m, rk.h)
  }

  
  rk.LinearCombination(m, rk.h, 1.0, h)
  //rk.linearCombinationMany(m.data, rk.kdata, rk.b, m.length)
  
  rk.time = time0 // will be incremented by simrun.go
}



func (rk *RK) String() string{
  str := ""
  for i:=0; i<rk.order; i++{
    str += fmt.Sprint(rk.c[i]) + "\t|\t"
    for j:=0; j<rk.order; j++{
      str += fmt.Sprint(rk.a[i][j]) + "\t"
    }
    str += "\n"
  }
  str += "----\n\t|\t"
  for i:=0; i<rk.order; i++{
    str += fmt.Sprint(rk.b[i]) + "\t"
  }
  return str
}