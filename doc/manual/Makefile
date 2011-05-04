all: 
	pdflatex -halt-on-error manual.tex
	doxygen pydoxygen.cfg
	make -C latex
	mv latex/refman.pdf ../../

.PHONY: clean
clean:
	rm -f *.aux *.bbl *.blg *.ind *.ilg *.log *.toc *.out *.pdf *.idx
