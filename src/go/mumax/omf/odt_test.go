package omf

import (
	"testing"
	"iotool"
)


func TestOdt(test *testing.T) {
	out := iotool.MustOpenWRONLY("test.odt")
	table := NewTable(out)
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
