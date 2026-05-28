#!/bin/bash
ITER_PER_PPM="90000"
NEIGHBORS="50"
PHOTONS_PER_ITER="4000000"

START_RADIUS="1.5"

SCENE="scene_massive_2.xml"
OUTFILE="gt_massive.exr"
python3 ../../test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS

SCENE="scene_big_2.xml"
OUTFILE="gt_big.exr"
python3 ../../test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS

SCENE="scene_2.xml"
OUTFILE="gt.exr"
python3 ../../test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS