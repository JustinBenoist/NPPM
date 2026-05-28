#!/bin/bash

ITER_PER_PPM="500000"
START_RADIUS="0.5"
NEIGHBORS="100"
PHOTONS_PER_ITER="2000000"
SCENE="scenes/scene_crab/scene.xml"
OUTFILE="scenes/scene_crab/gt_new.exr"

python3 test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --all_caustics
