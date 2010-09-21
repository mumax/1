all:
	simulate stdprobl4.in
	autoplot.sh plot

plot.pdf:	output/datatable.txt
	autoplot.sh plot

clean:
	rm -rf output *.eps *.svg *.pdf m*.png m*.t
