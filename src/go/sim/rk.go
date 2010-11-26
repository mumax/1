//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements a plethora of Runge-Kutta methods
// TODO: perhaps all the coefficients should be carefully
// double-checked for typos?
//
import (
	"fmt"
	"math"
)

// General Runge-Kutta solver
//
// y_(n+1) = y_(n) + dt * sum b_i * k_i
// k_i = torque(m0 + dt * sum a_ij * k_j, t0 + c_i*dt)
//
// butcher tableau:
//
// c0 | a00 a01 ...
// c1 | a10 a11 ...
// ...| ... ... ...
// ----------------
//    | b0  b1  ...
//    | b2_0 b2_1 ...
//
type RK struct {
	*Sim

	stages     int
	errororder float64 // the order of the less acurate solution used for the error estimate

	a     [][]float32
	b     []float32
	c     []float32
	b2    []float32    // weights to get lower order solution for error estimate, may be nil
	k     []*DevTensor //
	kdata []uintptr    //contiguous array of data pointers from k (kdata[i] = k[i].data), passable to C

	m0 *DevTensor // buffer to backup the starting magnetization

	Reductor // maxabs for error estimate 
}


// rk1: Euler's method
// 0 | 0
// -----
//   | 1
func NewRK1(sim *Sim) *RK {
	rk := newRK(sim, 1)
	rk.b[0] = 1.
	return rk
}


// rk2: Heun's method
// 0 | 0  0
// 1 | 1  0
// --------
//   |.5 .5
func NewRK2(sim *Sim) *RK {
	rk := newRK(sim, 2)
	rk.c[1] = 1.
	rk.a[1][0] = 1.
	rk.b[0] = .5
	rk.b[1] = .5
	return rk
}


// rk12: Adaptive Heun
// 0 | 0  0
// 1 | 1  0
// --------
//   |.5 .5
//   | 1  0
func NewRK12(sim *Sim) *RK {
	rk := NewRK2(sim)
	rk.initAdaptive(1.)
	rk.b2 = []float32{1., 0.}
	return rk
}


// rk3: Kutta's method
//  0  | 0    0  0
//  1/2| 1/2  0  0
//  1  | -1   2  0
// ----------------
//     | 1/6 2/3 1/6
func NewRK3(sim *Sim) *RK {
	rk := newRK(sim, 3)
	rk.c = []float32{0., 1. / 2., 1.}
	rk.a = [][]float32{
		{0., 0., 0.},
		{1. / 2., 0., 0.},
		{-1., 2., 0}}
	rk.b = []float32{1. / 6., 2. / 3., 1. / 6.}
	return rk
}


// rk23: Bogacki–Shampine method
func NewRK23(sim *Sim) *RK {
	rk := newRK(sim, 4)
	rk.c = []float32{0., 1. / 2., 3. / 4., 1.}
	rk.a = [][]float32{
		{0., 0., 0., 0.},
		{1. / 2., 0., 0., 0.},
		{0., 3. / 4., 0., 0.},
		{2. / 9., 1. / 3., 4. / 9., 0.}}
	rk.b = []float32{2. / 9., 1. / 3., 4. / 9., 0.}
	rk.initAdaptive(2.)
	rk.b2 = []float32{7. / 24, 1. / 4., 1. / 3., 1. / 8.}
	return rk
}


// rk4: The classical Runge-Kutta method
//  0  | 0    0  0   0
//  1/2| 1/2  0  0   0
//  1/2| 0  1/2  0   0
//  1  | 0    0  1   0
// ---------------------
//     | 1/6 1/3 1/3 1/6
func NewRK4(sim *Sim) *RK {
	rk := newRK(sim, 4)
	rk.c = []float32{0., 1. / 2., 1. / 2., 1.}
	rk.a = [][]float32{
		{0., 0., 0., 0.},
		{1. / 2., 0., 0., 0.},
		{0., 1. / 2., 0., 0.},
		{0., 0., 1., 0}}
	rk.b = []float32{1. / 6., 1. / 3., 1. / 3., 1. / 6.}
	return rk
}


// Cash-Karp
func NewRKCK(sim *Sim) *RK {
	rk := newRK(sim, 6)
	rk.c = []float32{0., 1. / 5., 3. / 10., 3. / 5., 1., 7. / 8.}
	rk.a = [][]float32{
		{0, 0, 0, 0, 0, 0},
		{1. / 5., 0, 0, 0, 0, 0},
		{3. / 40., 9. / 40., 0, 0, 0, 0},
		{3. / 10., -9. / 10., 6. / 5., 0, 0, 0},
		{-11. / 54., 5. / 2., -70. / 27., 35. / 27., 0, 0},
		{1631. / 55296., 175. / 512., 575. / 13824., 44275. / 110592., 253. / 4096., 0.}}
	rk.b = []float32{37. / 378., 0, 250. / 621., 125. / 594., 0, 512. / 1771.}
	rk.initAdaptive(4.)
	rk.b2 = []float32{2825. / 27648., 0, 18575. / 48384., 13525. / 55296., 277. / 14336., 1. / 4.}
	return rk
}

// Dormand-Prince
// Does not work yet
func NewRKDP(sim *Sim) *RK {
	rk := newRK(sim, 7)
	rk.c = []float32{0., 1. / 5., 3. / 10., 4. / 5., 8. / 9., 1., 1.}
	rk.a = [][]float32{
		{0., 0., 0., 0., 0., 0., 0.},
		{1. / 5., 0., 0., 0., 0., 0., 0.},
		{3. / 40., 9. / 40., 0., 0., 0., 0., 0.},
		{44. / 45., -56. / 15., 32. / 9., 0., 0., 0., 0.},
		{19372. / 6561., -25360. / 2187., 64448. / 6561., -212 / 729, 0., 0., 0.},
		{9017. / 3168., -355. / 33., 46732. / 5247., 49. / 176., -5103. / 18656., 0., 0.},
		{35. / 384., 0., 500. / 1113., 125. / 192., -2187. / 6784., 11. / 84., 0.}}
	rk.b = []float32{35. / 384., 0., 500. / 1113., 125. / 192., -2187. / 6784., 11. / 84., 0.}
	rk.initAdaptive(4.)
	rk.b2 = []float32{5179. / 57600., 0., 7571. / 16695., 393. / 640., -92097. / 339200., 187. / 2100., 1. / 40.}
	return rk
}

// INTERNAL
func (rk *RK) init(sim *Sim, order int) {

	rk.stages = order
	rk.Sim = sim

	rk.a = make([][]float32, order)
	for i := range rk.a {
		rk.a[i] = make([]float32, order)
	}
	rk.b = make([]float32, order)
	rk.c = make([]float32, order)
	rk.k = make([]*DevTensor, order)
	rk.kdata = make([]uintptr, order)
	for i := range rk.k {
		if i == 0 {
			// sim.h(Dev) is already allocated in sim but we don't really need it here,
			// therefore, h is "recycled" and used as k[0] to save memory.
			// Extra bonus: hDev now gets updated with (even accurate) values
			// of h/torque, so those can now also be outputted by sim.
			rk.k[i] = sim.h
		} else {
			rk.k[i] = NewTensor(sim.Backend, sim.mDev.size)
		}
		rk.kdata[i] = rk.k[i].data
	}
	rk.m0 = NewTensor(sim.Backend, sim.mDev.size)
}

func (rk *RK) initAdaptive(errororder float64) {
	rk.b2 = make([]float32, rk.stages)
	rk.Reductor.InitMaxAbs(rk.Sim.Backend, prod(rk.size4D[0:]))
	rk.errororder = errororder
}


// INTERNAL
func (rk *RK) free() {
	for i := range rk.k {
		rk.k[i].Free()
		rk.k[i] = nil
		rk.kdata[i] = 0
	}
}


// INTERNAL
func newRK(sim *Sim, order int) *RK {
	rk := new(RK)
	rk.init(sim, order)
	return rk
}


// INTERNAL
// func newAdaptiveRK(sim *Sim, order int) *RK {
// 	rk := newRK(sim, order)
// 	rk.initAdaptive()
// 	return rk
// }


func (rk *RK) Step() {

	order := rk.stages
	k := rk.k
	time0 := rk.time
	h := rk.dt
	c := rk.c
	m := rk.mDev
	m1 := rk.m0
	a := rk.a

	if rk.dt == 0. {
		rk.dt = 1e-5 // program units, should really be small enough
		rk.Println("Using default initial dt: ", rk.dt)
	}

	for i := 0; i < order; i++ {
		rk.time = time0 + float64(c[i]*h)
		TensorCopyOn(m, m1)
		for j := 0; j < order; j++ {
			if a[i][j] != 0. {
				rk.MAdd(m1, h*a[i][j], k[j])
			}
		}

		rk.calcHeff(m1, k[i])
		rk.Torque(m1, k[i])
	}

	//Lowest-order solution for error estimate, if applicable
	//from now, m1 = lower-order solution
	if rk.b2 != nil {
		TensorCopyOn(m, m1)
		for i := range k {
			if rk.b2[i] != 0. {
				rk.MAdd(m1, rk.b2[i]*h, k[i])
			}
		}
	}

	//Highest-order solution (m)
	//TODO: not 100% efficient, too many adds
	for i := range k {
		if rk.b[i] != 0. {
			rk.MAdd(m, rk.b[i]*h, k[i])
		}
	}

	//calculate error if applicable
	if rk.b2 != nil {
		rk.MAdd(m1, -1, m) // make difference between high- and low-order solution
		error := rk.Reduce(m1)
		rk.stepError = error

		// calculate new step
		assert(rk.maxError != 0.)
		//TODO: what is the pre-factor of the error estimate?
		factor := float32(math.Pow(float64(rk.maxError/error), 1./rk.errororder))

		// do not increase by time step by more than 100%
		if factor > 2. {
			factor = 2.
		}
		// do not decrease to less than 1%
		if factor < 0.01 {
			factor = 0.01
		}

		rk.dt = rk.dt * factor
	}

	//todo: undo bad steps

	rk.time = time0 // will be incremented by simrun.go
	rk.Normalize(m)
}


func (rk *RK) String() (str string) {
	defer func() { recover(); return }()

	for i := 0; i < rk.stages; i++ {
		str += fmt.Sprint(rk.c[i]) + "\t|\t"
		for j := 0; j < rk.stages; j++ {
			str += fmt.Sprint(rk.a[i][j]) + "\t"
		}
		str += "\n"
	}
	str += "----\n\t|\t"
	for i := 0; i < rk.stages; i++ {
		str += fmt.Sprint(rk.b[i]) + "\t"
	}
	if rk.b2 != nil {
		str += "\n"
		for i := 0; i < rk.stages; i++ {
			str += fmt.Sprint(rk.b2[i]) + "\t"
		}
	}
	return str
}
