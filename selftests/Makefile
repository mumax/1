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
	template solver=rk1,rk2,rk3,rk4 prob4fixdt
	template solver=rk12,rk23,rkdp,rkck prob4


