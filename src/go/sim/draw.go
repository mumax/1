package sim

import (
	"tensor"
	. "image"
	"image/png"
	// 	"os"
	// 	"fmt"
	"io"
)


// func main() {
// 	t := Read(os.Stdin)
// 	if Rank(t) != 2 {
// 		panic("tensor.Tensor should have rank 2")
// 	}
// 	PNG(os.Stdout, t)
// }


func PNG(out io.Writer, t tensor.Tensor) {
	err := png.Encode(out, DrawTensor(t))
	if err != nil {
		panic(err)
	}
}

func DrawTensor(t tensor.Tensor) *NRGBA {
	w, h := t.Size()[0], t.Size()[1]
	img := NewNRGBA(w, h)
	pos := []int{w, h}
	for i := 0; i < w; i++ {
		pos[X] = i
		for j := 0; j < h; j++ {
			pos[Y] = j
			img.Set(i, j, GreyMap(-1., 1., t.Get(pos)))
		}
	}
	return img
}

func GreyMap(min, max, value float) NRGBAColor {
	color := (value - min) / (max - min)
	if color > 1. {
		color = 1.
	}
	if color < 0. {
		color = 0.
	}
	color8 := uint8(255 * color)
	return NRGBAColor{color8, color8, color8, 255}
}
