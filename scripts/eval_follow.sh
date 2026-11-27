#!/bin/bash
# Evaluate each adapter as its training run finishes, while later runs train.
#
# Each pass evaluates whatever adapters results/train/summary.json lists and have no
# replies yet. The loop ends once scripts/train_all.sh has stopped and a pass that
# started after that has finished, and then writes the report. Extra arguments go
# to the eval step.

cd "$(dirname "$0")/.." || exit 2
while true; do
    stopped=no
    if [ -f runs/train/STOPPED ]; then stopped=yes; fi
    .venv/bin/python -u -m posttrain_lab eval --no-report "$@"
    code=$?
    rm -f runs/eval/ACTIVE
    if [ "$code" -eq 3 ] || [ "$code" -eq 2 ]; then exit "$code"; fi
    if [ "$stopped" = yes ] && [ "$code" -eq 0 ]; then break; fi
    sleep 60
done
.venv/bin/python -u -m posttrain_lab eval "$@"
