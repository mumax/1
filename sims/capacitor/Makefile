# Simulates a perpendicularly magnetized thin film
# Like the electrical field of a capacitor, H should be approximately 1

N0=128
N1=128
N2=8

all:
	config --size $(N0) $(N1) $(N2) --uniform 0 0 1 > m_init.t
	gosim --msat 800E3 --aexch 1.3e-11 --size $(N0) $(N1) $(N2) --cellsize 3.90625E-9 3.90625E-9 3E-9 --loadm m_init.t --dt 0. --steps 1 --storeh h.t
	tensor --read h.t --gnuplot > h.gnu
	tensor --read h.t --get Z 63 63 3

.PHONY:	clean
clean:
	rm -f h.gnu h.t m00*.t m_init.t
