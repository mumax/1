//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package common

import(
	"testing"
)

func TestIsFinite(t *testing.T){
	var one float32 = 1
	var zero float32 = 0

	if !IsReal(0.) {t.Fail()}
	if !IsReal(1.) {t.Fail()}
	if IsReal(zero/zero) {t.Fail()}
	if IsReal(one/zero) {t.Fail()}
	if IsReal(-one/zero) {t.Fail()}

	if IsFinite(0.) {t.Fail()}
	if !IsFinite(1.) {t.Fail()}
	if IsFinite(zero/zero) {t.Fail()}
	if IsFinite(one/zero) {t.Fail()}
	if IsFinite(-one/zero) {t.Fail()}
}
