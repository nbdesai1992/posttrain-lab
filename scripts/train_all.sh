#!/bin/bash
# Train every planned run, starting again after a failure.
#
# A run that fails stays pending, so each pass trains only what is left. The loop
# stops when nothing is left, on an account problem, or after six passes. Extra
# arguments go to the train step, for example --config configs/train-pilot.toml.

cd "$(dirname "$0")/.." || exit 2
mkdir -p runs/train
rm -f runs/train/STOPPED
code=1
for pass in 1 2 3 4 5 6; do
    .venv/bin/python -u -m posttrain_lab train "$@"
    code=$?
    if [ "$code" -ne 1 ]; then break; fi
    sleep 300
done
echo "$code" > runs/train/STOPPED    # tells scripts/eval_follow.sh that no more adapters are coming
exit "$code"
