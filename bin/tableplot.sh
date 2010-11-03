#! /bin/bash
#
# @file
# This script automates plotting with gnuplot.
#
#
svgrc='set term svg size 900 600 fixed fname "Helvetica" fsize 22 butt;'
epsrc='set term post eps color solid font "Helvetica" 20 linewidth 3.0;'



for i; do
	echo $i;
  file=\"$i\"

  cmd_m='set xlabel "time(ns)"; set ylabel "magnetization"; plot '$file' using ($1*1E9):2 with lines title "mx", '$file' using ($1*1E9):3 with lines title "my", '$file' using ($1*1E9):4 with lines title "mz";'

  epsfile=$i.m.eps
  pdffile=$i.m.pdf
  svgfile=$i.m.svg

  echo $(echo $epsrc; echo set output '"'$epsfile'";'; echo $cmd_m; echo set output';') | gnuplot;
  echo $(echo $svgrc; echo set output '"'$svgfile'";'; echo $cmd_m; echo set output';') | gnuplot;
  ps2pdf -dEPSCrop $epsfile $pdffile

  cmd_b='set xlabel "time(ns)"; set ylabel "external field (T)"; plot '$file' using ($1*1E9):5 with lines title "Bx", '$file' using ($1*1E9):6 with lines title "By", '$file' using ($1*1E9):7 with lines title "Bz";'

  epsfile=$i.b.eps
  pdffile=$i.b.pdf
  svgfile=$i.b.svg

  echo $(echo $epsrc; echo set output '"'$epsfile'";'; echo $cmd_b; echo set output';') | gnuplot;
  echo $(echo $svgrc; echo set output '"'$svgfile'";'; echo $cmd_b; echo set output';') | gnuplot;
  ps2pdf -dEPSCrop $epsfile $pdffile

  cmd_b='set xlabel "time(ns)"; set ylabel "max torque(unitless)"; plot '$file' using ($1*1E9):8 with lines title "tx", '$file' using ($1*1E9):9 with lines title "ty", '$file' using ($1*1E9):10 with lines title "tz";'

  epsfile=$i.torque.eps
  pdffile=$i.torque.pdf
  svgfile=$i.torque.svg

  echo $(echo $epsrc; echo set output '"'$epsfile'";'; echo $cmd_b; echo set output';') | gnuplot;
  echo $(echo $svgrc; echo set output '"'$svgfile'";'; echo $cmd_b; echo set output';') | gnuplot;
  ps2pdf -dEPSCrop $epsfile $pdffile


done;
