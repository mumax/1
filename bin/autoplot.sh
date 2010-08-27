#! /bin/bash
#
# @file
# This script automates plotting with gnuplot.
# 
# Gnuplot commands are stored in a file typically named 'plot', 
# which could contatin, e.g.,:
#
# set xlabel "time";
# plot "file.txt";
#
# (note that the commands must be semicolon-separated)
# This script will store the corresponding plot in eps, pdf and svg format.
# 
# @author: Arne Vansteenkiste
#
svgrc='set term svg size 900 600 fixed fname "Helvetica" fsize 24 butt;'
epsrc='set term post eps color solid font "Helvetica" 28 linewidth 3.0;'

for i; do
	echo $i;
	epsfile=$i.eps
	pdffile=$i.pdf
	svgfile=$i.svg
	echo $(echo $epsrc; echo set output '"'$epsfile'";'; cat $i; echo set output';') | gnuplot;
	echo $(echo $svgrc; echo set output '"'$svgfile'";'; cat $i; echo set output';') | gnuplot;
	ps2pdf -dEPSCrop $epsfile
done;
