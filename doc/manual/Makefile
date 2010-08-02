all:	manual.pdf

manual.pdf:	*.tex	#manual.bbl
	pdflatex -halt-on-error manual.tex
	pdflatex -halt-on-error manual.tex

# manual.bbl:	*.bib
# 	pdflatex -halt-on-error manual.tex
# 	bibtex main
# 	pdflatex -halt-on-error manual.tex

.PHONY: clean
clean:
	rm -f *.aux *.bbl *.blg *.ind *.ilg *.log *.toc *.out *.idx
