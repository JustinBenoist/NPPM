#!/bin/bash

ITER_PER_PPM="100000"
START_RADIUS="1.2"
NEIGHBORS="100"
PHOTONS_PER_ITER="1000000"
SCENE="scenes/glass_thinlens_2/glass_SPPM.xml"
OUTFILE="scenes/glass_thinlens_2/gt_new.exr"

python3 test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0
