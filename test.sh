#!/bin/bash

# ITER_PER_PPM="200"
ITER_PER_PPM="500000"
NEIGHBORS="50"
PHOTONS_PER_ITER="8000000"

DCV_SIZE="32"
# MODEL="output/training_local_caustic_r/model.pth"

# DCV_SIZE="32"
MODEL="output/model.pth"
# MODEL="output/training_agressive_4_dcv32_clampratio/model.pth"
ENCODER="output/encoder.pth"

START_RADIUS="0.5"
# SCENE="scenes/mitsuba_CPPM_scenes/artware/artware_SPPM.xml"
SCENE="scenes/classroom/scene_v3.xml"
# SCENE="scenes/scene_crab/scene.xml"
# SCENE="scenes/mitsuba_CPPM_scenes/box/box.xml"

# START_RADIUS="0.05"
# SCENE="scenes/water-caustic_glossy/scene_v0.6.xml"
# OUTFILE="output/training_local_caustic/ours_opt_water_g.exr"

STOP_GRID="50"
RES_GRID="256"
TIME_LIMIT="30"

# REF="scenes/mitsuba_CPPM_scenes/artware/gt_caustics.exr"
# ERROR="/home/jbenoist/Documents/figures_NPPM/fwsdfs.pkl"
# REF="scenes/scene_crab_thinlens/gt_caustics.exr"
# ERROR="output/pretrain_encoder/DoF/pretrain_beta/crab_thinlens/error.pkl"
# --ref $REF --error_out $ERROR


# python3 test.py --k 0.8 --beta 1.2 --min_photon 10 --cut_radius 10000 --ref $REF --error_out $ERROR --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --model $MODEL --dcv_size $DCV_SIZE --avg --cppm
# python3 test.py --k 0.8 --beta 1.2 --min_photon 10 --ref $REF --error_out $ERROR --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --cppm
# python3 test.py --ref $REF --error_out $ERROR --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --all_caustics --cppm

OUTFILE="scenes/classroom/trash.exr"
# OUTFILE="scenes/classroom/ours_150it.exr"
python3 test.py --encoder $ENCODER --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --model $MODEL --dcv_size $DCV_SIZE --stop_grid $STOP_GRID --res_grid $RES_GRID --time_limit $TIME_LIMIT
# OUTFILE="output/equal-time-new/crab_15/trash.exr"
# python3 test.py --encoder $ENCODER --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --model $MODEL --dcv_size $DCV_SIZE --stop_grid $STOP_GRID --res_grid $RES_GRID --cppm --all_caustics
# python3 test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --model $MODEL --dcv_size $DCV_SIZE --stop_grid $STOP_GRID --res_grid $RES_GRID --all_caustics --cppm
# OUTFILE="output/equal-time-new/crab_15/trash.exr"
# python3 test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --cppm --all_caustics
# OUTFILE="output/equal-time-new/crab_15/trash.exr"
# OUTFILE="scenes/classroom/gt_glossy.exr"
# python3 test.py --scene $SCENE --outfile $OUTFILE --iter $ITER_PER_PPM --radius $START_RADIUS --ppi $PHOTONS_PER_ITER --neighbors $NEIGHBORS --seed 0 --time_limit $TIME_LIMIT --cppm