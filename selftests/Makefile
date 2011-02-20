test:	gpu cpu

gpu:	clean templates
	mumax  -daemon -watch=0
	maxtest

cpu:	clean templates
	mumax -cpu  -daemon -watch=0
	maxtest

clean:
	rm -rf *.out

templates:
	template solver=rk1,rk2,rk3,rk4 prob4fixdt.in.template
	template solver=rk12,rk23,rkdp,rkck prob4.in.template
	template dt=0,1e-17,1e-12,1e-10 mindm=1e-3 maxdm=1e-1 minmaxdm.in.template
	template re-init.in.template
	#template relax.in.template
	template anis.in.template
