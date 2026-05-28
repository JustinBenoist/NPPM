#!/bin/bash

ITER_PER_PPM="2000000"
START_RADIUS="0.334"
NEIGHBORS="100"
PHOTONS_PER_ITER="1000000"
SCENE="scenes/mitsuba_CPPM_scenes/artware/artware_SPPM.xml"
OUTFILE="scenes/mitsuba_CPPM_scenes/artware/gt_new.exr"

python3 test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0
