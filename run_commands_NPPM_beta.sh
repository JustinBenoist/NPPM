# Scene: glass_thinlens
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/glass_thinlens_2/glass_SPPM.xml" --ref "scenes/glass_thinlens_2/gt_caustics.exr" --outfile "output/convergence/NPPM_beta/glass_thinlens_2/NPPM_beta.exr" --error_out "output/convergence/NPPM_beta/glass_thinlens_2/error_NPPM_beta.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 23523 --cppm

# Scene: glass
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/glass/glass_SPPM.xml" --ref "scenes/glass/gt_caustics.exr" --outfile "output/convergence/NPPM_beta/glass/NPPM_beta.exr" --error_out "output/convergence/NPPM_beta/glass/error_NPPM_beta.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 23523 --cppm

# Scene: crab
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/scene_crab/scene.xml" --ref "scenes/scene_crab/gt_caustics.exr" --outfile "output/convergence/NPPM_beta/crab/NPPM_beta.exr" --error_out "output/convergence/NPPM_beta/crab/error_NPPM_beta.pkl" --iter 100000 --radius 6.0 --ppi 2000000 --neighbors 50 --seed 0 --all_caustics --cppm

# Scene: artware
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/artware/artware_SPPM.xml" --ref "scenes/artware/gt_caustics.exr" --outfile "output/convergence/NPPM_beta/artware/NPPM_beta.exr" --error_out "output/convergence/NPPM_beta/artware/error_NPPM_beta.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 0 --cppm --time_limit 30

# Scene: box
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/box/box.xml" --ref "scenes/box/gt_caustics.exr" --outfile "output/convergence/NPPM_beta/box/NPPM_beta.exr" --error_out "output/convergence/NPPM_beta/box/error_NPPM_beta.pkl" --iter 100000 --radius 6.0 --ppi 400000 --neighbors 50 --seed 0 --cut_radius 4.0 --cppm

# Scene: crab_thinlens
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/scene_crab_thinlens/scene.xml" --ref "scenes/scene_crab_thinlens/gt_coral_2.exr" --outfile "output/convergence/NPPM_beta/crab_thinlens/NPPM_beta.exr" --error_out "output/convergence/NPPM_beta/crab_thinlens/error_NPPM_beta.pkl" --iter 100000 --radius 6.0 --ppi 2000000 --neighbors 50 --seed 0 --all_caustics --cppm --time_limit 30

# Scene: classroom
python3 test.py --encoder "model/encoder.pth" --model "model/model.pth" --scene "scenes/classroom/scene_v3.xml" --ref "scenes/classroom/gt_caustics.exr" --outfile "output/convergence/NPPM_beta/classroom/NPPM_beta.exr" --error_out "output/convergence/NPPM_beta/classroom/error_NPPM_beta.pkl" --iter 1000 --radius 3.0 --ppi 2000000 --neighbors 50 --seed 0 --cppm --time_limit 300
