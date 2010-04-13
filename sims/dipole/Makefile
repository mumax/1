# Simulates a perpendicularly magnetized thin film
# Like the electrical field of a capacitor, H should be approximately 1

N0=64
N1=64
N2=1

all:
	tensor --new 3 $(N0) $(N1) $(N2) --setall 1. > m_init.t
	gosim --msat 800E3 --aexch 1.3e-11 --size $(N0) $(N1) $(N2) --cellsize 10e-9 10e-9 10e-9 --loadm m_init.t --dt 0. --steps 1 --storeh h.t
	tensor --read h.t --normalize 0 --gnuplot > h.gnu
	vectorplot.sh h.t

.PHONY:	clean
clean:
	rm -f h.gnu h.t m00*.t m_init.t
