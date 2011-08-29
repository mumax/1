//  This file is part of MuMax, a high-performance micromagnetic simulator.
//  Copyright 2010  Arne Vansteenkiste
//  Use of this source code is governed by the GNU General Public License version 3
//  (as published by the Free Software Foundation) that can be found in the license.txt file.
//  Note that you are welcome to modify this code under the condition that you do not remove any 
//  copyright notices and prominently state that you modified it, giving a relevant date.

// This file implements version checking

package common

import (
	"http"
	"fmt"
	"os"
	"io/ioutil"
	"strconv"
	"strings"
)


// Read a simple text file on the webserver to see if a newer version is available
func CheckVersion(url string, myversion int) (shouldUpgrade bool) {
	defer func() {
		err := recover()
		if err != nil {
			fmt.Fprintln(os.Stderr, err) // TODO: rm
		}
	}()
	println("Hit ", url)
	var client http.Client
	resp, _, err := client.Get(url)

	if err != nil {
		fmt.Fprintln(os.Stderr, err) // TODO: rm
		return                       // too bad
	}

	fmt.Println(resp)

	if resp.StatusCode == 200 { // OK
		bodybuf, err2 := ioutil.ReadAll(resp.Body)
		if err2 != nil {
			fmt.Fprintln(os.Stderr, err) // TODO: rm
			return                       // too bad
		}
		body := strings.Trim(string(bodybuf), " \n\t")
		println(body)
		server, err3 := strconv.Atoi(body)
		if err3 != nil {
			fmt.Fprintln(os.Stderr, err3) // TODO: rm
			return                       // too bad
		}
		shouldUpgrade = server > myversion
	}
	return
}
