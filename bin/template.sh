#! /bin/bash
#
# Utility to generate .h/.cu pair from a template
#
# usage:
# template.sh filename
# will create filename.h, filename.cu with
# basic contents
#
# @author Arne Vansteenkiste 

file=$1

(
echo $file'.o:  '$file.h $file.cu
echo -e '$(NVCC) -c $(CUDALIBS) $(NVCCFLAGS) '$file.cu
)


if [ -e $file.h ]; then
  echo $file.h exits;
  exit;
fi

if [ -e $file.cu ]; then
  echo $file.cu exits;
  exit;
fi

(
echo '/**'
echo ' * @file'
echo ' *'
echo ' * @author Arne Vansteenkiste'
echo ' * @author Ben Van de Wiele'
echo ' */'
echo '#ifndef '$file'_h'
echo '#define '$file'_h'
echo 
echo '#ifdef __cplusplus'
echo 'extern "C" {'
echo '#endif'
echo
echo
echo
echo '#ifdef __cplusplus'
echo '}'
echo '#endif'
echo '#endif'
)>$file.h


(
echo '#include "'$file.h'"'
echo 
echo '#ifdef __cplusplus'
echo 'extern "C" {'
echo '#endif'
echo
echo
echo
echo '#ifdef __cplusplus'
echo '}'
echo '#endif'
)>$file.cu


kate $file.h $file.cu Makefile