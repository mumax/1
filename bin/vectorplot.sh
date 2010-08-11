#! /bin/bash
tensor --gnuplot < $1 > $1.gnu
cmd="set term png size 800,800; set view 0,0 ; set output \"$1.png\"; splot \"$1.gnu\" with vectors; set output; exit";
echo $cmd
echo $cmd | gnuplot;
convert -crop 600x600+100+100 $1.png /tmp/tmp.png
mv /tmp/tmp.png $1.png

