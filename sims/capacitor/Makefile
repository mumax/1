# Simulates a perpendicularly magnetized thin film
# Like the electrical field of a capacitor, H should be approximately 1

N0=32
N1=32
N2=1

all:	m_init.t
	gosim --alpha 0.02  --msat 800E3 --aexch 1.3e-11 --size $(N0) $(N1) $(N2) --cellsize 3.90625E-9 3.90625E-9 3E-9 --loadm m_init.t --dt 0.1E-12 --steps 10000 --autosavem 1E-12

m_init.t:
	tensor --new 3 $(N0) $(N1) $(N2) --setall 1.0 --normalize 0 > m0.t



kernel.t:
	kernel --size $(N0) $(N1) $(N2) --cellsize 3.90625E-9 3.90625E-9 3E-9 --msat 800E3 --aexch 1.3E-11 > kernel.t

.PHONY:	clean
clean:
	rm -f m00*.t kernel.t *.png
