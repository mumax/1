//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

import (
	. "testing"
	"reflect"
	//. "os";
)

/*
 * Tensor Unit Tests
 */

func TestIO(t *T) {
	A := NewTensor3([]int{5, 7, 9})
	for i := range A.List() {
		A.List()[i] = float(i) / 10.0
	}
	Write(FOpenz("iotest"), A)
	A2 := Read(FOpenz("iotest"))
	//Print(Stdout, A2);
	if !Equals(A, A2) {
		t.Fail()
	}
}

func TestTensorOps(t *T) {

	// test SetAll and iteration
	a := NewTensor3([]int{2, 3, 4})
	SetAll(a, 0.5)
	count := 0
	for i := NewIterator(a); i.HasNext(); i.Next() {
		if a.Get(i.Index()) != 0.5 {
			t.Fail()
		}
		count++
	}
	if count != 3*4*2 {
		t.Fail()
	}

	b := Copy(a)
	if !Equals(a, b) {
		t.Fail()
	}

	c := Buffer(a)
	if !Equals(a, c) {
		t.Fail()
	}

	// test Slicing
	s1 := Slice(a, 0, 1)
	if !reflect.DeepEqual(s1.Size(), []int{3, 4}) {
		t.Fail()
	}
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if a.Array()[0][i][j] != s1.Get([]int{i, j}) {
				t.Fail()
			}
		}
	}

	s2 := Slice(a, 1, 2)
	if !reflect.DeepEqual(s2.Size(), []int{2, 4}) {
		t.Fail()
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 4; j++ {
			if a.Array()[0][i][j] != s1.Get([]int{i, j}) {
				t.Fail()
			}
		}
	}

	s3 := Slice(a, 2, 0)
	if !reflect.DeepEqual(s3.Size(), []int{2, 3}) {
		t.Fail()
	}
	s4 := Slice(s3, 0, 1)
	if !reflect.DeepEqual(s4.Size(), []int{3}) {
		t.Fail()
	}

	// test Transpose
	t1 := Transpose(a, 0, 1)
	if !reflect.DeepEqual(t1.Size(), []int{3, 2, 4}) {
		t.Fail()
	}

}
//
// func testData(){
//
//   const(N0 = 3;
//         N1 = 4;
// 	N2 = 5
//   );
//
//   a := NewVectorBlock([]int{N0, N1, N2});
//   b := NewVectorBlock([]int{N0, N1, N2});
//
//   assert(a.Equals(b));
//
//   a.Set(1, 2, 3, 3.14);
//   assert(a.Get3(1, 2, 3) == 3.14);
//   assert(!a.Equals(b));
//
//   for i:= range(a.AsArray()){
//     a.AsArray()[i] = float(i);
//   }
//
//   Copy(a, b);
//   assert(a.Equals(b));
//
//   c := a.Clone();
//   assert(a.Equals(c));
//
//   d:= NewVectorBlock([]int{2*N0, 3*N1, 4*N2});
//   a.CopyInto(d);
//   for i:=0; i<a.Size0(); i++{
//     for j:=0; j<a.Size1(); j++{
//       for k:=0; k<a.Size2(); k++{
// 	if(a.Get3(i,j,k) != d.Get3(i,j,k)){
// 	  assert(false);
// 	}
//       }
//     }
//   }
//   assert(true);
//
// }

func TestComponent(t *T) {
	a := NewTensor4([]int{3, 5, 5, 2})
	b := a.Component(1)
	if Rank(b) != 3 {
		t.Fail()
	}
	if N(b) != 5*5*2 {
		t.Fail()
	}
	SetAll(b, 1.23)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			for k := 0; k < 2; k++ {
				if a.Get([]int{0, i, j, k}) != 0. {
					t.Fail()
				}
				if a.Get([]int{1, i, j, k}) != 1.23 {
					t.Fail()
				}
				if a.Get([]int{2, i, j, k}) != 0. {
					t.Fail()
				}
			}
		}
	}

	b = a.Component(2)
	//   if Rank(b) != 3 { t.Fail() };
	//   if N(b) != 5*5*2 { t.Fail() };
	//   SetAll(b, 3.45);
	//   for i:=0; i<5; i++{
	//     for j:=0; j<5; j++{
	//       for k:=0; k<2; k++{
	// 	if a.Get([]int{0, i, j, k}) != 0. { t.Fail() };
	// 	if a.Get([]int{1, i, j, k}) != 1.23 { t.Fail() };
	// 	if a.Get([]int{2, i, j, k}) != 3.45 { t.Fail() };
	//       }
	//     }
	//   }
}

func TestComponentN(t *T) {
	a := NewTensorN([]int{3, 5, 5, 2})
	b := a.Component(1)
	if Rank(b) != 3 {
		t.Fail()
	}
	if N(b) != 5*5*2 {
		t.Fail()
	}
	SetAll(b, 1.23)
	for i := 0; i < 5; i++ {
		for j := 0; j < 5; j++ {
			for k := 0; k < 2; k++ {
				if a.Get([]int{0, i, j, k}) != 0. {
					t.Fail()
				}
				if a.Get([]int{1, i, j, k}) != 1.23 {
					t.Fail()
				}
				if a.Get([]int{2, i, j, k}) != 0. {
					t.Fail()
				}
			}
		}
	}
}

func TestTensor0(t *T) {
	a := NewTensor0()

	if Rank(a) != 0 {
		t.Fail()
	}

	arr := a.List()

	arr[0] = 123.
	if a.Get([]int{}) != 123. {
		t.Fail()
	}

}

func TestTensor1(t *T) {
	a := NewTensor1([]int{3})

	if Rank(a) != 1 {
		t.Fail()
	}
	if a.Size()[0] != 3 {
		t.Fail()
	}

	arr := a.Array()
	for i := 0; i < 3; i++ {
		arr[i] = float(i)
	}

	for i := 0; i < 3; i++ {
		if a.Get([]int{i}) != float(i) {
			t.Fail()
		}
	}

}


func TestTensor2(t *T) {
	a := NewTensor2([]int{3, 4})

	if Rank(a) != 2 {
		t.Fail()
	}
	if a.Size()[0] != 3 {
		t.Fail()
	}
	if a.Size()[1] != 4 {
		t.Fail()
	}

	a.List()[11] = 1.23
	if a.Array()[2][3] != 1.23 {
		t.Fail()
	}
	if a.Get([]int{2, 3}) != 1.23 {
		t.Fail()
	}

	arr := a.Array()
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			arr[i][j] = float(i + 2*j)
		}
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			if a.Get([]int{i, j}) != float(i+2*j) {
				t.Fail()
			}
		}
	}

}


func TestTensor3(t *T) {
	a := NewTensor3([]int{3, 4, 5})

	if Rank(a) != 3 {
		t.Fail()
	}
	if a.Size()[0] != 3 {
		t.Fail()
	}
	if a.Size()[1] != 4 {
		t.Fail()
	}
	if a.Size()[2] != 5 {
		t.Fail()
	}

	arr := a.Array()
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 5; k++ {
				arr[i][j][k] = float(i + 2*j + 3*k)
			}
		}
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 5; k++ {
				if a.Get([]int{i, j, k}) != float(i+2*j+3*k) {
					t.Fail()
				}
			}
		}
	}
}


func TestTensor4(t *T) {
	a := NewTensor4([]int{3, 4, 5, 6})

	if Rank(a) != 4 {
		t.Fail()
	}
	if a.Size()[0] != 3 {
		t.Fail()
	}
	if a.Size()[1] != 4 {
		t.Fail()
	}
	if a.Size()[2] != 5 {
		t.Fail()
	}
	if a.Size()[3] != 6 {
		t.Fail()
	}

	arr := a.Array()
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 5; k++ {
				for l := 0; l < 6; l++ {
					arr[i][j][k][l] = float(i + 2*j + 3*k + 4*l)
				}
			}
		}
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 5; k++ {
				for l := 0; l < 6; l++ {
					if a.Get([]int{i, j, k, l}) != float(i+2*j+3*k+4*l) {
						t.Fail()
					}
				}
			}
		}
	}

}

func TestTensor5(t *T) {
	a := NewTensor5([]int{3, 4, 5, 6, 7})

	if Rank(a) != 5 {
		t.Fail()
	}
	if a.Size()[0] != 3 {
		t.Fail()
	}
	if a.Size()[1] != 4 {
		t.Fail()
	}
	if a.Size()[2] != 5 {
		t.Fail()
	}
	if a.Size()[3] != 6 {
		t.Fail()
	}
	if a.Size()[4] != 7 {
		t.Fail()
	}

	arr := a.Array()
	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 5; k++ {
				for l := 0; l < 6; l++ {
					for m := 0; m < 7; m++ {
						arr[i][j][k][l][m] = float(i + 2*j + 3*k + 4*l + 5*m)
					}
				}
			}
		}
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 5; k++ {
				for l := 0; l < 6; l++ {
					for m := 0; m < 7; m++ {
						if a.Get([]int{i, j, k, l, m}) != float(i+2*j+3*k+4*l+5*m) {
							t.Fail()
						}
					}
				}
			}
		}
	}
}

func TestTensorN(t *T) {
	a := NewTensorN([]int{3, 4, 5, 6, 7})

	if Rank(a) != 5 {
		t.Fail()
	}
	if a.Size()[0] != 3 {
		t.Fail()
	}
	if a.Size()[1] != 4 {
		t.Fail()
	}
	if a.Size()[2] != 5 {
		t.Fail()
	}
	if a.Size()[3] != 6 {
		t.Fail()
	}
	if a.Size()[4] != 7 {
		t.Fail()
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 5; k++ {
				for l := 0; l < 6; l++ {
					for m := 0; m < 7; m++ {
						Set(a, []int{i, j, k, l, m}, float(i+2*j+3*k+4*l+5*m))
					}
				}
			}
		}
	}

	for i := 0; i < 3; i++ {
		for j := 0; j < 4; j++ {
			for k := 0; k < 5; k++ {
				for l := 0; l < 6; l++ {
					for m := 0; m < 7; m++ {
						if a.Get([]int{i, j, k, l, m}) != float(i+2*j+3*k+4*l+5*m) {
							t.Fail()
						}
					}
				}
			}
		}
	}

}
