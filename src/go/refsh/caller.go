//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package refsh

import (
	. "reflect"
)

// Caller unifies anything that can be called:
// a Method or a FuncValue
type Caller interface {
	// Call the thing
	Call(args []Value) []Value
	// Types of the input parameters
	In(i int) Type
	// Number of input parameters
	NumIn() int
}


type MethodWrapper struct {
	reciever Value
	function *FuncValue
}

func (m *MethodWrapper) Call(args []Value) []Value {
	methargs := make([]Value, len(args)+1) // todo: buffer in method struct
	methargs[0] = m.reciever
	for i, arg := range args {
		methargs[i+1] = arg
	}
	return m.function.Call(methargs)
}

func (m *MethodWrapper) In(i int) Type {
	return (m.function.Type().(*FuncType)).In(i + 1) // do not treat the reciever (1st argument) as an actual argument
}

func (m *MethodWrapper) NumIn() int {
	return (m.function.Type().(*FuncType)).NumIn() - 1 // do not treat the reciever (1st argument) as an actual argument
}


type FuncWrapper FuncValue

func (f *FuncWrapper) In(i int) Type {
	return (*FuncValue)(f).Type().(*FuncType).In(i)
}

func (f *FuncWrapper) NumIn() int {
	return (*FuncValue)(f).Type().(*FuncType).NumIn()
}

func (f *FuncWrapper) Call(args []Value) []Value {
	return (*FuncValue)(f).Call(args)
}
