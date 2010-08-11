#! /bin/bash

# Generates bin/setenv

OUTPUT=bin/sim
echo '#! /bin/bash' > $OUTPUT
echo 'export SIMROOT='$(pwd) >> $OUTPUT
cat src/setenv_32bit.template >> $OUTPUT
chmod u+x $OUTPUT

echo Created $OUTPUT.
echo You can now run this script to start simulation programs.