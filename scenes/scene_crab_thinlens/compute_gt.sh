#!/bin/bash

ITER_PER_PPM="100000"
START_RADIUS="1.2"
NEIGHBORS="100"
PHOTONS_PER_ITER="2000000"
SCENE="scenes/scene_crab_thinlens/scene.xml"
OUTFILE="scenes/scene_crab_thinlens/gt_caustics.exr"

python3 test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --all_caustics
