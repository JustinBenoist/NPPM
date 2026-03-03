#!/bin/bash

ITER_PER_PPM="200000"
START_RADIUS="0.02"
NEIGHBORS="100"
PHOTONS_PER_ITER="1000000"
SCENE="scenes/water-caustic_glossy/scene_v0.6.xml"
OUTFILE="scenes/water-caustic_glossy/gt_new.exr"

python3 test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0
