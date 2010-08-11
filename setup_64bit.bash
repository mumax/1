#! /bin/bash

# Generates bin/setenv
# TODO we need to quote simroot

OUTPUT=bin/sim
echo '#! /bin/bash' > $OUTPUT
echo 'export SIMROOT='$(pwd) >> $OUTPUT
cat src/setenv_64bit.template >> $OUTPUT
chmod u+x $OUTPUT

echo Created $OUTPUT.
echo You can now run this script to start simulation programs.