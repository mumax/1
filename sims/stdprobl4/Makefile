# Standard problem 4
# @see: http://www.ctcms.nist.gov/~rdm/mumag.org.html
#

N0=128
N1=32
N2=1

all:	m0.t kernel.t
	gpusim_test m0.t kernel.t
# (for i in m00*.t; do tensor --component X --average 0 --average 0 --get 0 < $i; done) > mx.txt

m0.t:
	tensor --new 3 $(N0) $(N1) $(N2) --setall 1.0 --normalize 0 > m0.t

kernel.t:
	kernel --size $(N0) $(N1) $(N2) --cellsize 3.90625E-9 3.90625E-9 3E-9 --msat 800E3 --aexch 1.3E-11 > kernel.t
#	kernel --size $(N0) $(N1) $(N2) --cellsize 1.953125E-9 1.953125E-9 1.5E-9 --msat 800E3 --aexch 1.3E-11 > kernel.t

.PHONY:	clean
clean:
	rm -f m00*.t m0.t kernel.t *.png
