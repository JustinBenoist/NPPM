# Scene: glass_thinlens
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/glass_thinlens_2/glass_SPPM.xml" --ref "scenes/glass_thinlens_2/gt_caustics.exr" --outfile "output/convergence/NPPM_alpha/glass_thinlens_2/NPPM_alpha.exr" --error_out "output/convergence/NPPM_alpha/glass_thinlens_2/error_NPPM_alpha.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 23523

# Scene: glass
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/glass/glass_SPPM.xml" --ref "scenes/glass/gt_caustics.exr" --outfile "output/convergence/NPPM_alpha/glass/NPPM_alpha.exr" --error_out "output/convergence/NPPM_alpha/glass/error_NPPM_alpha.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 23523

# Scene: crab
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/scene_crab/scene.xml" --ref "scenes/scene_crab/gt_caustics.exr" --outfile "output/convergence/NPPM_alpha/crab/NPPM_alpha.exr" --error_out "output/convergence/NPPM_alpha/crab/error_NPPM_alpha.pkl" --iter 100000 --radius 6.0 --ppi 2000000 --neighbors 50 --seed 0 --all_caustics

# Scene: artware
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/artware/artware_SPPM.xml" --ref "scenes/artware/gt_caustics.exr" --outfile "output/convergence/NPPM_alpha/artware/NPPM_alpha.exr" --error_out "output/convergence/NPPM_alpha/artware/error_NPPM_alpha.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 0 --time_limit 30

# Scene: box
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/box/box.xml" --ref "scenes/box/gt_caustics.exr" --outfile "output/convergence/NPPM_alpha/box/NPPM_alpha.exr" --error_out "output/convergence/NPPM_alpha/box/error_NPPM_alpha.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 0 --cut_radius 4.0


# Scene: crab_thinlens
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/scene_crab_thinlens/scene.xml" --ref "scenes/scene_crab_thinlens/gt_coral_2.exr" --outfile "output/convergence/NPPM_alpha/crab_thinlens/NPPM_alpha.exr" --error_out "output/convergence/NPPM_alpha/crab_thinlens/error_NPPM_alpha.pkl" --iter 100000 --radius 6.0 --ppi 2000000 --neighbors 50 --seed 0 --all_caustics --time_limit 30

# Scene: classroom
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/classroom/scene_v3.xml" --ref "scenes/classroom/gt_caustics.exr" --outfile "output/convergence/NPPM_alpha/classroom/NPPM_alpha.exr" --error_out "output/convergence/NPPM_alpha/classroom/error_NPPM_alpha.pkl" --iter 1000 --radius 3.0 --ppi 2000000 --neighbors 50 --seed 0 --time_limit 300
