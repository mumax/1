all:
	pdflatex -halt-on-error manual.tex
	makeindex manual

# manual.bbl:	*.bib
# 	pdflatex -halt-on-error manual.tex
# 	bibtex main
# 	pdflatex -halt-on-error manual.tex

.PHONY: clean
clean:
	rm -f *.aux *.bbl *.blg *.ind *.ilg *.log *.toc *.out *.pdf *.idx
