#! /bin/bash
#
# Utility to find-and-replace in all source files at once.
#
# usage:
# refactor.sh "searchtext" "replacetext"
# will replace "searchtext" by "replacetext" 
# in all *.h *.c *.cpp *.cu files in the
# working directory
#
# @author Arne Vansteenkiste 

grep --color $1 *.h *.c *.cpp *.cu *.go;

for i in *.h *.c *.cpp *.cu *.go; do
  NUM=$(grep $1 $i | wc -l)
  if (( $NUM > 0 )); then
    cat $i | sed 's/'$1'/'$2'/g' > refactor-tmp;
    mv refactor-tmp $i;
  fi;
done;

rm -f refactor-tmp