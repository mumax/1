package refsh

import (
	"strings"
)

//TODO overloading, abbreviations, ...
func (r *Refsh) resolve(funcname string) Caller {
	funcname = strings.ToLower(funcname) // be case-insensitive
	for i := range r.funcnames {
		if r.funcnames[i] == funcname {
			return r.funcs[i]
		}
	}
	return nil
}
