# Scene: glass_thinlens
python3 test.py --scene "scenes/glass_thinlens_2/glass_SPPM.xml" --ref "scenes/glass_thinlens_2/gt_caustics.exr" --outfile "output/convergence/CPPM/glass_thinlens_2/CPPM.exr" --error_out "output/convergence/CPPM/glass_thinlens_2/error_CPPM.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 42 --cppm

# Scene: glass
python3 test.py --scene "scenes/glass/glass_SPPM.xml" --ref "scenes/glass/gt_caustics.exr" --outfile "output/convergence/CPPM/glass/CPPM.exr" --error_out "output/convergence/CPPM/glass/error_CPPM.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 42 --cppm

# Scene: crab
python3 test.py --scene "scenes/scene_crab/scene.xml" --ref "scenes/scene_crab/gt_caustics.exr" --outfile "output/convergence/CPPM/crab/CPPM.exr" --error_out "output/convergence/CPPM/crab/error_CPPM.pkl" --iter 100000 --radius 6.0 --ppi 2000000 --neighbors 50 --seed 0 --all_caustics --cppm

# Scene: artware
python3 test.py --scene "scenes/artware/artware_SPPM.xml" --ref "scenes/artware/gt_caustics.exr" --outfile "output/convergence/CPPM/artware/CPPM.exr" --error_out "output/convergence/CPPM/artware/error_CPPM.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 0 --cppm --time_limit 30

# Scene: box
python3 test.py --scene "scenes/box/box.xml" --ref "scenes/box/gt_caustics.exr" --outfile "output/convergence/CPPM/box/CPPM.exr" --error_out "output/convergence/CPPM/box/error_CPPM.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 0 --cppm

# Scene: crab_thinlens
python3 test.py --scene "scenes/scene_crab_thinlens/scene.xml" --ref "scenes/scene_crab_thinlens/gt_coral_2.exr" --outfile "output/convergence/CPPM/crab_thinlens/CPPM.exr" --error_out "output/convergence/CPPM/crab_thinlens/error_CPPM.pkl" --iter 100000 --radius 6.0 --ppi 2000000 --neighbors 50 --seed 0 --all_caustics --cppm --time_limit 30

# Scene: classroom
python3 test.py --scene "scenes/classroom/scene_v3.xml" --ref "scenes/classroom/gt_caustics.exr" --outfile "output/convergence/CPPM/classroom/CPPM.exr" --error_out "output/convergence/CPPM/classroom/error_CPPM.pkl" --iter 1000 --radius 3.0 --ppi 2000000 --neighbors 50 --seed 0 --cppm --time_limit 300

