//  Copyright 2010  Arne Vansteenkiste, Ben Van de Wiele
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package sim

// This file implements methods for generating
// initial magnetization configurations like
// vortices, Landau patterns, etc.

import (
	. "mumax/common"
	"rand"
	"mumax/omf"
	"math"
)


func (sim *Sim) SetMRange(z1, y1, x1, z2, y2, x2 int, mz, my, mx float32) {
	sim.initMLocal()
	s := sim.mLocal.Array()
	for i := x1; i <= x2; i++ {
		for j := y1; j <= y2; j++ {
			for k := z1; k <= z2; k++ {
				s[X][i][j][k] = mx
				s[Y][i][j][k] = my
				s[Z][i][j][k] = mz
			}
		}
	}
}

// Sets the magnetization of cell with integer index (x,y,z) to (mx, my, mz)
func (s *Sim) SetMCell(z, y, x int, mz, my, mx float32) {
	s.initMLocal()
	a := s.mLocal.Array()
	a[X][x][y][z] = mx
	a[Y][x][y][z] = my
	a[Z][x][y][z] = mz
	s.invalidate() // todo: we do not need to invalidate everything here!
}


func (s *Sim) SetM(z, y, x float32, mz, my, mx float32) {
	s.initMLocal()
	i := int((x/s.input.cellSize[X])*float32(s.input.size[X]) - (s.input.partSize[X] / 2.) + 0.5)
	j := int((y/s.input.cellSize[Y])*float32(s.input.size[Y]) - (s.input.partSize[Y] / 2.) + 0.5)
	k := int((z/s.input.cellSize[Z])*float32(s.input.size[Z]) - (s.input.partSize[Z] / 2.) + 0.5)
	s.SetMCell(i, j, k, mx, my, mz)
}

// Make the magnetization uniform.
// (mx, my, mz) needs not to be normalized.
func (s *Sim) Uniform(mz, my, mx float32) {
	s.initMLocal()
	a := s.mLocal.Array()
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {
				a[X][i][j][k] = mx
				a[Y][i][j][k] = my
				a[Z][i][j][k] = mz
			}
		}
	}
	s.invalidate() // todo: we do not need to invalidate everything here!
}

// Make a vortex with given
// in-plane circulation (-1 or +1)
// and core polarization (-1 or 1)
func (s *Sim) Vortex(circulation, polarization int) {
	s.initMLocal()
	cy, cx := s.size[1]/2, s.size[2]/2
	a := s.mLocal.Array()
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][0][j] {
				y := j - cy
				x := k - cx
				a[X][i][j][k] = 0
				a[Y][i][j][k] = float32(x * circulation)
				a[Z][i][j][k] = float32(-y * circulation)
			}
		}
		a[Z][i][cy][cx] = 0.
		a[Y][i][cy][cx] = 0.
		a[X][i][cy][cx] = float32(polarization)
	}
	s.invalidate()
}

// Make part of vortex-array with given
// bottom left coordinate [x1, y1] and top right coordinate [x1,y1], (expressed in #cells)
// vortex separation D (expressed in #cells)
// in-plane circulation (-1 or +1)
// and core polarization (-1 or 1)
// The data array is expected to have only zeros at the start
func (s *Sim) Vortex_in_array(vrtx_i, vrtx_j, unit_size, separation float32, circulation, polarization int) {
	s.initMLocal()
	c := s.input.cellSize
	B := unit_size / c[2]
	S := separation / c[2]

	cx, cy := int((vrtx_i+0.5)*B), int((vrtx_j+0.5)*B)
	a := s.mLocal.Array()
	for i := range a[0] {
		for j := int(vrtx_j*B + S/2); j < int((vrtx_j+1)*B-S/2); j++ {
			for k := int(vrtx_i*B + S/2); k < int((vrtx_i+1)*B-S/2); k++ {
				y := j - cy
				x := k - cx
				a[X][i][j][k] = 0
				a[Y][i][j][k] = float32(x * circulation)
				a[Z][i][j][k] = float32(-y * circulation)
			}
		}
		a[Z][i][cy][cx] = 0.
		a[Y][i][cy][cx] = 0.
		a[X][i][cy][cx] = float32(polarization)
	}
	s.invalidate()
}


func (s *Sim) VortexInArray(vrtx_i, vrtx_j, unit_size_x, unit_size_y, separation_x, separation_y float32, circulation, polarization int) {
  s.initMLocal()
  c := s.input.cellSize
  Bx := unit_size_x / c[2]
  Sx := separation_x / c[2]
  By := unit_size_y / c[1]
  Sy := separation_y / c[1]

// cx := int((vrtx_i+0.5)*Bx) 
  cx := float32(vrtx_i)*Bx + Bx/2 - 0.5
  cy := float32(vrtx_j)*By + By/2 - 0.5
/*  cy :=int((vrtx_j+0.5)*By)*/
  a := s.mLocal.Array()
  for i := range a[0] {
/*    for j := int(vrtx_j*By + Sy/2); j < int((vrtx_j+1)*By-Sy/2); j++ {
      for k := int(vrtx_i*Bx + Sx/2); k < int((vrtx_i+1)*By-Sy/2); k++ {*/
    for j := int(float32(vrtx_j)*By + Sy/2); j<int(float32(vrtx_j+1)*By - Sy/2); j++ {
      for k := int(float32(vrtx_i)*Bx + Sx/2); k<int(float32(vrtx_i+1)*Bx - Sx/2); k++ {
        y := float32(j) - cy
        x := float32(k) - cx
        a[X][i][j][k] = 0
        a[Y][i][j][k] = x*float32(circulation)
        a[Z][i][j][k] = -y*float32(circulation)
      }
    }
    a[Z][i][int(cy)][int(cx)] = 0.
    a[Y][i][int(cy)][int(cx)] = 0.
    a[X][i][int(cy)][int(cx)] = float32(polarization)
  }
  s.invalidate()
}

func (s *Sim) SetVortexEllips(cx, cy, sx, sy float32, circulation, polarization int){
  s.initMLocal()
  c := s.input.cellSize
  cxint := int(cx/c[2])
  cyint := int(cy/c[1])
  rxint := int(sx/c[2]/2)
  ryint := int(sy/c[1]/2)

  a := s.mLocal.Array()
  for i := range a[0] {
    for j := -ryint; j<ryint + 1; j++ {
      for k := -rxint; k<rxint + 1; k++ {
        if ( float32(j*j)/float32(ryint*ryint) + float32(k*k)/float32(rxint*rxint) <= 1.0 ){
          a[X][i][j+cyint][k+cxint] = 0
          a[Y][i][j+cyint][k+cxint] = float32( k*circulation)
          a[Z][i][j+cyint][k+cxint] = float32(-j*circulation)
        }
      }
    }
    a[Z][i][cyint][cxint] = 0.
    a[Y][i][cyint][cxint] = 0.
    a[X][i][cyint][cxint] = float32(polarization)
  }
  s.invalidate()
}


func (s *Sim) SetVortexRectangle(cx, cy, sx, sy float32, circulation, polarization int){
  s.initMLocal()
  c := s.input.cellSize
  cxint := int(cx/c[2])
  cyint := int(cy/c[1])
  rxint := int(sx/c[2]/2)
  ryint := int(sy/c[1]/2)

  a := s.mLocal.Array()
  for i := range a[0] {
    for j := -ryint; j<ryint + 1; j++ {
      for k := -rxint; k<rxint + 1; k++ {
        a[X][i][j+cyint][k+cxint] = 0
        a[Y][i][j+cyint][k+cxint] = float32( k*circulation)
        a[Z][i][j+cyint][k+cxint] = float32(-j*circulation)
      }
    }
    a[Z][i][cyint][cxint] = 0.
    a[Y][i][cyint][cxint] = 0.
    a[X][i][cyint][cxint] = float32(polarization)
  }
  s.invalidate()
}



// Make the magnetization the starting configuration of a symmetric Bloch wall
func (s *Sim) SBW() {
	s.initMLocal()
	a := s.mLocal.Array()
	pi := math.Pi
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {
				theta := pi * (float64(k-s.size[2]/2.0) + .5) / float64(s.size[2])
				phi := 0.0
				a[X][i][j][k] = float32(math.Sin(theta))
				a[Y][i][j][k] = float32(math.Cos(theta) * math.Cos(phi))
				a[Z][i][j][k] = float32(math.Cos(theta) * math.Sin(phi))
			}
		}
	}
	s.invalidate()
}


// Make the magnetization the starting configuration of a symmetric Néel wall
func (s *Sim) SNW() {
	s.initMLocal()
	a := s.mLocal.Array()
	pi := math.Pi
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {
				theta := pi * (float64(k-s.size[2]/2.0) + .5) / float64(s.size[2])
				phi := pi / 2.0
				a[X][i][j][k] = float32(math.Sin(theta))
				a[Y][i][j][k] = float32(math.Cos(theta) * math.Cos(phi))
				a[Z][i][j][k] = float32(math.Cos(theta) * math.Sin(phi))
			}
		}
	}
	s.invalidate()
}


// Make the magnetization the starting configuration of a asymmetric Bloch wall
func (s *Sim) ABW() {
	s.initMLocal()
	a := s.mLocal.Array()
	pi := math.Pi
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {
				theta := pi * (float64(k-s.size[2]/2.0) + .5) / float64(s.size[2])
				phi := pi * (float64(j-s.size[1]/2.0) + .5) / float64(s.size[1])
				a[X][i][j][k] = float32(math.Sin(theta))
				a[Y][i][j][k] = float32(math.Cos(theta) * math.Cos(phi))
				a[Z][i][j][k] = float32(math.Cos(theta) * math.Sin(phi))
			}
		}
	}
	s.invalidate()
}


// Make the magnetization the starting configuration of a asymmetric Néel wall
func (s *Sim) ANW() {
	s.initMLocal()
	a := s.mLocal.Array()
	pi := math.Pi
	for i := range a[0] {
		for j := range a[0][i] {
			for k := range a[0][i][j] {
				theta := pi * (float64(k-s.size[2]/2.0) + .5) / float64(s.size[2])
				phi := pi / 4.0
				a[X][i][j][k] = float32(math.Sin(theta))
				a[Y][i][j][k] = float32(math.Cos(theta) * math.Cos(phi))
				a[Z][i][j][k] = float32(math.Cos(theta) * math.Sin(phi))
			}
		}
	}
	s.invalidate()
}


func (s *Sim) LoadM(file string) {
	s.initMLocal()
	s.Println("Loading ", file)
	_, s.mLocal = omf.FRead(file) // omf.Info is discarded for now, could be used for scaling.
	//TODO this should not invalidate the entire sim
	s.invalidate()
}


// Adds noise with the specified amplitude
// to the magnetization state.
// Handy to break the symmetry.
func (s *Sim) AddNoise(amplitude float32) {
	s.initMLocal()
	amplitude *= 2
	list := s.mLocal.List()
	for i := range list {
		list[i] += amplitude * float32(rand.Float32()-0.5)
	}
	s.invalidate()
}

// Sets the initial magnetization random
func (s *Sim) SetRandom() {
	s.initMLocal()
	list := s.mLocal.List()
	for i := range list {
		list[i] = float32(rand.Float32() - 0.5)
	}
	s.invalidate()
}


// Sets the random seed
func (s *Sim) RandomSeed(seed int64) {
	rand.Seed(seed)
}


// TODO: we are in trouble here if we have automatic transpose of the geometry for performance
// X needs to be the out-of-plane direction
