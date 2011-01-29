//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

package device

import(
	. "mumax/common"
//	"mumax/device/cpu"
	"io/ioutil"
	"strings"
)

// For testing purposes only:
// file where we save "cpu", "gpu", ... defining
// the device to be used by "make test". 
// In this way we can set a device from the makefile.
const TEST_DEVICE_FILE = "/tmp/mumax_test_device"

// To be included as the first line of every TestXXX() func.
// Reads /tmp/mumax_test_device and sets the device according
// to its contents ("cpu", "gpu", "multigpu", ...)
func InitDevice() {
	bytes, err := ioutil.ReadFile(TEST_DEVICE_FILE)
	dev := strings.ToLower(string(bytes))
	CheckErr(err, ERR_IO)
	switch{
		default:	panic(InputErr(TEST_DEVICE_FILE + ": " + dev))
		case strings.HasPrefix(dev, "cpu"):
			Use(nil)//cpu.Init(cpu.MaxThreads(), 0))
		case strings.HasPrefix(dev, "gpu"):
		case strings.HasPrefix(dev, "multigpu"):
	}
}


