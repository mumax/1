# Standard problem 4
# @see: http://www.ctcms.nist.gov/~rdm/mumag.org.html
# field 1: µ0Hx=-24.6 mT, µ0Hy= 4.3 mT, µ0Hz= 0.0 mT
# 
# We test for non-cubic cells.
#

N0=1
N1=64
N2=128
X=3.90625E-9
Y=1.953125E-9
Z=3E-9

all:	m_init.t
	gosim --alpha 0.02  --msat 800E3 --aexch 1.3e-11 --size $(N0) $(N1) $(N2) --cellsize $(Z) $(Y) $(X) --loadm m_init.t --dt 0.1E-12 --hext 0 4.3E-3 -24.6E-3 --autosavem 1E-12 --steps 10000
	make plot
	make movie

plot:	plot.pdf

plot.pdf:
	bash -c '(for i in m00*.t; do tensor --component Y --average 0 --average 0 --get 0 < $$i; done) > my.txt'
	autoplot.sh plot

movie:	movie.avi

movie.avi:
	bash -c 'for i in m*.t; do tensor --read $$i --component Z --transpose X Z  --slice Z 0 | draw > $$i.png; done; movie.sh *.png'
m_init.t:
	tensor --new 3 $(N0) $(N1) $(N2) --setall 1.0 --normalize 0 > m0.t
	gosim --alpha 1.0 --msat 800E3 --aexch 1.3e-11 --size $(N0) $(N1) $(N2) --cellsize $(Z) $(Y) $(X) --loadm m0.t --dt 0.5E-12 --steps 4000 --storem m_init.t

.PHONY:	clean
clean:
	rm -f m00*.t kernel.t *.png m0.t m_init.t plot.pdf plot.svg plot.eps my.txt movie.avi *.png

