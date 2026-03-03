#!/bin/bash
ITER_PER_PPM="500"
NEIGHBORS="50"
PHOTONS_PER_ITER="4000000"

START_RADIUS="12.0"
SCENE="scene.xml"

DCV_SIZE="32"
MODEL="../../output/model.pth"
ENCODER="../../output/encoder.pth"
STOP_GRID="50"
RES_GRID="64"
TIME_LIMIT="30"

# OUTFILE="test_ppm.exr"
# python3 ../../test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS

OUTFILE="test_nppm_64.exr"
python3 ../../test.py --encoder $ENCODER --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --model $MODEL --dcv_size $DCV_SIZE --stop_grid $STOP_GRID --res_grid $RES_GRID