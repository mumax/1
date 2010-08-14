package refsh

//TODO overloading, abbreviations, ...
func (r *Refsh) resolve(funcname string) Caller {
	for i := range r.funcnames {
		if r.funcnames[i] == funcname {
			return r.funcs[i]
		}
	}
	return nil
}
