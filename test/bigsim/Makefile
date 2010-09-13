all:
	simulate bigsim.in

plot.pdf: output/datatable.txt
	autoplot.sh plot.txt

clean:
	rm -rf output *.eps *.svg *.pdf
