test:	gpu cpu

gpu:	clean templates
	mumax -silent -daemon -watch=0
	maxtest

cpu:	clean templates
	mumax -cpu -silent -daemon -watch=0
	maxtest

clean:
	rm -rf *.out

templates:
	template solver=rk1,rk2,rk12,rk23 prob4


