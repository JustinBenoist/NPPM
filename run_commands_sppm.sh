# Scene: glass_thinlens
python3 test.py --scene "scenes/glass_thinlens_2/glass_SPPM.xml" --ref "scenes/glass_thinlens_2/gt_caustics.exr" --outfile "output/convergence/SPPM/glass_thinlens_2/SPPM.exr" --error_out "output/convergence/SPPM/glass_thinlens_2/error_SPPM.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 42

# Scene: glass
python3 test.py --scene "scenes/glass/glass_SPPM.xml" --ref "scenes/glass/gt_caustics.exr" --outfile "output/convergence/SPPM/glass/SPPM.exr" --error_out "output/convergence/SPPM/glass/error_SPPM.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 42

# Scene: crab
python3 test.py --scene "scenes/scene_crab/scene.xml" --ref "scenes/scene_crab/gt_caustics.exr" --outfile "output/convergence/SPPM/crab/SPPM.exr" --error_out "output/convergence/SPPM/crab/error_SPPM.pkl" --iter 100000 --radius 6.0 --ppi 2000000 --neighbors 50 --seed 0 --all_caustics

# Scene: artware
python3 test.py --scene "scenes/artware/artware_SPPM.xml" --ref "scenes/artware/gt_caustics.exr" --outfile "output/convergence/SPPM/artware/SPPM.exr" --error_out "output/convergence/SPPM/artware/error_SPPM.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 0 --time_limit 30

# Scene: box
python3 test.py --scene "scenes/box/box.xml" --ref "scenes/box/gt_caustics.exr" --outfile "output/convergence/SPPM/box/SPPM.exr" --error_out "output/convergence/SPPM/box/error_SPPM.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 0

# Scene: crab_thinlens
python3 test.py --scene "scenes/scene_crab_thinlens/scene.xml" --ref "scenes/scene_crab_thinlens/gt_coral_2.exr" --outfile "output/convergence/SPPM/crab_thinlens/SPPM.exr" --error_out "output/convergence/SPPM/crab_thinlens/error_SPPM.pkl" --iter 100000 --radius 6.0 --ppi 2000000 --neighbors 50 --seed 0 --all_caustics --time_limit 30

# Scene: classroom
python3 test.py --scene "scenes/classroom/scene_v3.xml" --ref "scenes/classroom/gt_caustics.exr" --outfile "output/convergence/SPPM/classroom/SPPM.exr" --error_out "output/convergence/SPPM/classroom/error_SPPM.pkl" --iter 1000 --radius 3.0 --ppi 2000000 --neighbors 50 --seed 0 --time_limit 300