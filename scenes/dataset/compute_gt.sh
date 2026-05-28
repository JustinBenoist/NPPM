#!/bin/bash

ITER_PER_PPM="15000"
START_RADIUS="0.04"
NEIGHBORS="100"
PHOTONS_PER_ITER="400000"
SCENE="scenes/better_dataset_2/scene_$1.xml"
OUTFILE="scenes/better_dataset_2/gt_$1.exr"

python3 test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0
