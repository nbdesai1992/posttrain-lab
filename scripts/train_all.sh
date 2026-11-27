#!/bin/bash
# Train every planned run, one process per run, starting again after a failure.
#
# The config caps the runs a process may train, because the service makes a third
# training client from one process wait in its queue for a long time. Each pass
# trains what the cap allows and leaves the rest pending. The loop stops when
# nothing is left, on an account problem, or after five passes in a row that
# finished no run. Extra arguments go to the train step, for example
# --config configs/train-pilot.toml.

cd "$(dirname "$0")/.." || exit 2
mkdir -p runs/train
rm -f runs/train/STOPPED
finished() { grep -c . runs/train/results.jsonl 2>/dev/null || echo 0; }
code=1
idle=0
while [ "$idle" -lt 5 ]; do
    before=$(finished)
    .venv/bin/python -u -m posttrain_lab train "$@"
    code=$?
    if [ "$code" -ne 1 ]; then break; fi
    if [ "$(finished)" -gt "$before" ]; then
        idle=0
    else
        idle=$((idle + 1))
        sleep 300
    fi
done
echo "$code" > runs/train/STOPPED    # tells scripts/eval_follow.sh that no more adapters are coming
exit "$code"
