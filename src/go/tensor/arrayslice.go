//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

/*
 * Allocates multi-dimensional arrays of float's,
 * backed by contiguous one-dimensional arrays.
 * Such arrays can be passed to FFTW, yet they
 * are accessible with[][][] .
 */


/** Returns a 2D array, as well as the contiguous 1D array backing it. */

func Array2D(size0, size1 int) ([]float, [][]float) {
	assertMsg(size0 > 0 && size1 > 0, "Array size must be > 0")

	/* First make the slice and then the list. When the memory is not fragmented,
	 * they are probably allocated in a good order for the cache.
	 */
	sliced := make([][]float, size0)
	list := make([]float, size0*size1)
	CheckAlignment(list)

	for i := 0; i < size0; i++ {
		sliced[i] = list[i*size1 : (i+1)*size1]
	}
	return list, sliced
}

/** Returns a 3D array, as well as the contiguous 1D array backing it. */

func Array3D(size0, size1, size2 int) ([]float, [][][]float) {

	/* First make the slices and then the list. When the memory is not fragmented,
	 * they are probably allocated in a good order for the cache.
	 */
	sliced := make([][][]float, size0)
	for i := range sliced {
		sliced[i] = make([][]float, size1)
	}
	list := make([]float, size0*size1*size2)
	CheckAlignment(list)

	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = list[(i*size1+j)*size2+0 : (i*size1+j)*size2+size2]
		}
	}
	return list, sliced
}


/** Returns a 4D array, as well as the contiguous 1D array backing it. */

func Array4D(size0, size1, size2, size3 int) ([]float, [][][][]float) {

	/* First make the slices and then the list. When the memory is not fragmented,
	 * they are probably allocated in a good order for the cache.
	 */
	sliced := make([][][][]float, size0)
	for i := range sliced {
		sliced[i] = make([][][]float, size1)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = make([][]float, size2)
		}
	}
	list := make([]float, size0*size1*size2*size3)
	CheckAlignment(list)

	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				sliced[i][j][k] = list[((i*size1+j)*size2+k)*size3+0 : ((i*size1+j)*size2+k)*size3+size3]
			}
		}
	}
	return list, sliced
}


/** Returns a 5D array, as well as the contiguous 1D array backing it. */

func Array5D(size0, size1, size2, size3, size4 int) ([]float, [][][][][]float) {

	/* First make the slices and then the list. When the memory is not fragmented,
	 * they are probably allocated in a good order for the cache.
	 */
	sliced := make([][][][][]float, size0)
	for i := range sliced {
		sliced[i] = make([][][][]float, size1)
	}
	for i := range sliced {
		for j := range sliced[i] {
			sliced[i][j] = make([][][]float, size2)
		}
	}
	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				sliced[i][j][k] = make([][]float, size3)
			}
		}
	}
	list := make([]float, size0*size1*size2*size3*size4)
	CheckAlignment(list)

	for i := range sliced {
		for j := range sliced[i] {
			for k := range sliced[i][j] {
				for l := range sliced[i][j][k] {
					sliced[i][j][k][l] = list[(((i*size1+j)*size2+k)*size3+l)*size4+0 : (((i*size1+j)*size2+k)*size3+l)*size4+size4]
				}
			}
		}
	}
	return list, sliced
}
