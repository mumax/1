//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package tensor

/*
int pointer_to_int(void* ptr){
#ifdef _64_BIT
  return (long long int) ptr;
#else
  return (long int) ptr;
#endif
}

void array_copy(void* source, void* dest, int length){
 int i;
 float* s = (float*) source;	// it works for all 32-bit data types
 float* d = (float*) dest;
 for(i=0; i<length; i++){
  d[i] = s[i];
 }
}

*/
import "C"
import "unsafe"

/** Converts an int to a slice of 4 bytes, using the machine's endianess. */
func IntToBytes(i int, bytes []byte) {
	i32 := int32(i) //just to be sure...
	source := unsafe.Pointer(&i32)
	dest := unsafe.Pointer(&(bytes[0]))
	C.array_copy(source, dest, 1)
}

/** Converts a slice of 4 bytes to an int, using the machine's endianess. */
func BytesToInt(bytes []byte) int {
	// idea: check len(bytes)
	var i int
	dest := unsafe.Pointer(&i)
	source := unsafe.Pointer(&(bytes[0]))
	C.array_copy(source, dest, 1)
	return i
}

/** Converts a float to a slice of 4 bytes, using the machine's endianess. */
func FloatToBytes(i float, bytes []byte) {
	source := unsafe.Pointer(&i)
	dest := unsafe.Pointer(&(bytes[0]))
	C.array_copy(source, dest, 1)
}

/** Converts a slice of 4 bytes to a float, using the machine's endianess. */
func BytesToFloat(bytes []byte) float {
	var i float
	dest := unsafe.Pointer(&i)
	source := unsafe.Pointer(&(bytes[0]))
	C.array_copy(source, dest, 1)
	return i
}

/** Used internally to check alignments. */
func ToInt(pointer unsafe.Pointer) int64 {
	return (int64)(C.pointer_to_int(pointer))
}
