package omf

import (
	. "mumax/common"
	"testing"
)


func TestWrite(test *testing.T) {
	out := MustOpenWRONLY("test.odt")
	table := NewTabWriter(out)
	table.Title = "test"
	table.AddColumn("Mx", "A/m")
	table.AddColumn("My", "A/m")
	table.AddColumn("Mz", "A/m")
	table.AddColumn("E", "J")
	table.Print(0.95, 0.1, 0)
	table.Print(17)
	table.Print(0.96, 0.1, 0)
	table.Print(16)
	table.Close()
}

func TestRead(test *testing.T) {
	in := MustOpenRDONLY("test.odt")
	table := ReadTable(in)
	mx := table.GetColumn("Mx")
	if mx[0] != 0.95 {
		test.Fail()
	}
	if mx[1] != 0.96 {
		test.Fail()
	}
}
