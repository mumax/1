all: manual.bbl
	doxygen pydoxygen.cfg
	pdflatex -halt-on-error manual.tex
	makeindex manual

manual.bbl:	*.bib
	pdflatex -halt-on-error manual.tex
	bibtex manual
	pdflatex -halt-on-error manual.tex

.PHONY: clean
clean:
	rm -f *.aux *.bbl *.blg *.ind *.ilg *.log *.toc *.out *.pdf *.idx
