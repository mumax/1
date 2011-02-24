// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//  This code was taken from sort.go and modified by Arne Vansteenkiste, Feb 24, 2011
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.

package common

import (
	"sort"
)

// Float32Array attaches the methods of Interface to []float32, sorting in increasing order.
type Float32Array []float32

func (p Float32Array) Len() int           { return len(p) }
func (p Float32Array) Less(i, j int) bool { return p[i] < p[j] }
func (p Float32Array) Swap(i, j int)      { p[i], p[j] = p[j], p[i] }

// Sort is a convenience method.
func (p Float32Array) Sort() { sort.Sort(p) }
