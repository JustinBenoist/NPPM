#!/bin/bash

ITER_PER_PPM="10"
START_RADIUS="4"
NEIGHBORS="150"
NB_PPM="399"
PHOTONS_PER_ITER="400000"
SCENE="scenes/swimming_pool/scene.xml"
OUTFILE="scenes/swimming_pool/gt_caustics.exr"

python3 test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --stochastic

for i in $(seq 1 $NB_PPM);
do  
    START=$((ITER_PER_PPM*i))
    SEED=$((PHOTONS_PER_ITER*i))
    python3 test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --checkpoint $OUTFILE --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --start $START --seed $SEED --stochastic
done