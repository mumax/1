
#solver
solvertype  	heun
maxerror 	1E-5

# geometry
include geom.in

#initial magnetization
load 		relaxed.t


# excite
autosave	table	ascii	 50E-12
autosave	m	png	250E-12

