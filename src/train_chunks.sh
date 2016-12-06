#!/bin/bash
for i in {1..1}; do let seed=($i+2016); echo $i; echo $seed; for n in {1..26}; do echo $n;python xgb_training_v0.py -s $seed -i '../input/features/chunk'$n'.csv' -m '../models/xgb_chunk_'$i'_'$n'_v0.bin' -p '../models/xgb_chunk_'$i'_'$n'_val_v0.csv';done;done
