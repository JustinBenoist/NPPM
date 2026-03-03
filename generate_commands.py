import os

BASE_DIR = "output/convergence/NPPM_beta"
METHOD_NAME = "NPPM_beta"  # Adjust if needed

# Defaults from your bash
DEFAULT_ITER_PER_PPM = 1000
DEFAULT_START_RADIUS = 12.0
DEFAULT_NEIGHBORS = 50
DEFAULT_PHOTONS_PER_ITER = 400000
DEFAULT_DCV_SIZE = 32

MODEL_PATH = f"output/convergence/{METHOD_NAME}/model.pth"
ENCODER_PATH = f"output/convergence/{METHOD_NAME}/encoder.pth"

for scene_dir_name in sorted(os.listdir(BASE_DIR)):
    scene_dir = os.path.join(BASE_DIR, scene_dir_name)
    test_sh = os.path.join(scene_dir, "test.sh")

    if not os.path.isdir(scene_dir) or not os.path.isfile(test_sh):
        continue

    scene_name = scene_dir_name

    SCENE_XML = f"scenes/mitsuba_CPPM_scenes/{scene_name}/artware_SPPM.xml"
    REFERENCE_IMG = f"scenes/mitsuba_CPPM_scenes/{scene_name}/gt_caustics.exr"

    OUTDIR = f"output/convergence/{METHOD_NAME}/{scene_name}"
    OUTPUT_IMG = f"{OUTDIR}/{METHOD_NAME}.exr"
    ERROR_OUT = f"{OUTDIR}/error_{METHOD_NAME}.pkl"

    cmd = (
        f"python3 test.py "
        f"--encoder \"{ENCODER_PATH}\" "
        f"--model \"{MODEL_PATH}\" "
        f"--dcv_size {DEFAULT_DCV_SIZE} "
        f"--scene \"{SCENE_XML}\" "
        f"--ref \"{REFERENCE_IMG}\" "
        f"--outfile \"{OUTPUT_IMG}\" "
        f"--error_out \"{ERROR_OUT}\" "
        f"--iter {DEFAULT_ITER_PER_PPM} "
        f"--radius {DEFAULT_START_RADIUS} "
        f"--ppi {DEFAULT_PHOTONS_PER_ITER} "
        f"--neighbors {DEFAULT_NEIGHBORS} "
        f"--seed 0"
    )

    print(f"# Scene: {scene_name}")
    print(cmd)
    print()
