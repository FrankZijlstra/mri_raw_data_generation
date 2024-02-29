# Reconstruction experiments: Baseline
python run.py -p ./experiments/fastmri_data_recon.txt ./experiments/20.txt ./experiments/fastmri_recon.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_20s_run1 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon.txt ./experiments/20.txt ./experiments/fastmri_recon.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_20s_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon.txt ./experiments/50.txt ./experiments/fastmri_recon.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_50s_run1 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon.txt ./experiments/50.txt ./experiments/fastmri_recon.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_50s_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon.txt ./experiments/100.txt ./experiments/fastmri_recon.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_100s_run1 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon.txt ./experiments/100.txt ./experiments/fastmri_recon.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_100s_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon.txt ./experiments/160.txt ./experiments/fastmri_recon.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_160s_run1 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon.txt ./experiments/160.txt ./experiments/fastmri_recon.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_160s_run1 -gpu 0 -seed 0

# Reconstruction experiments: Baseline with data augmentation
python run.py -p ./experiments/fastmri_data_recon.txt ./experiments/20.txt ./experiments/fastmri_recon_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_20s_aug_run1 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon.txt ./experiments/20.txt ./experiments/fastmri_recon_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_20s_aug_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon.txt ./experiments/50.txt ./experiments/fastmri_recon_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_50s_aug_run1 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon.txt ./experiments/50.txt ./experiments/fastmri_recon_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_50s_aug_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon.txt ./experiments/100.txt ./experiments/fastmri_recon_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_100s_aug_run1 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon.txt ./experiments/100.txt ./experiments/fastmri_recon_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_100s_aug_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon.txt ./experiments/160.txt ./experiments/fastmri_recon_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_160s_aug_run1 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon.txt ./experiments/160.txt ./experiments/fastmri_recon_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_160s_aug_run1 -gpu 0 -seed 0

# Reconstruction experiments: Synthetic data
python run.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/20.txt ./experiments/fastmri_recon_gen.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_20s_gen_run1 -g multi 0.75 -g nraw 20 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/20.txt ./experiments/fastmri_recon_gen.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_20s_gen_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/50.txt ./experiments/fastmri_recon_gen.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_50s_gen_run1 -g multi 0.75 -g nraw 20 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/50.txt ./experiments/fastmri_recon_gen.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_50s_gen_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/100.txt ./experiments/fastmri_recon_gen.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_100s_gen_run1 -g multi 0.75 -g nraw 20 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/100.txt ./experiments/fastmri_recon_gen.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_100s_gen_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/160.txt ./experiments/fastmri_recon_gen.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_160s_gen_run1 -g multi 0.75 -g nraw 20 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/160.txt ./experiments/fastmri_recon_gen.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_160s_gen_run1 -gpu 0 -seed 0

# Reconstruction experiments: Synthetic data + data augmentation
python run.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/20.txt ./experiments/fastmri_recon_gen_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_20s_gen_aug_run1 -g multi 0.75 -g nraw 20 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/20.txt ./experiments/fastmri_recon_gen_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_20s_gen_aug_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/50.txt ./experiments/fastmri_recon_gen_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_50s_gen_aug_run1 -g multi 0.75 -g nraw 20 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/50.txt ./experiments/fastmri_recon_gen_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_50s_gen_aug_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/100.txt ./experiments/fastmri_recon_gen_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_100s_gen_aug_run1 -g multi 0.75 -g nraw 20 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/100.txt ./experiments/fastmri_recon_gen_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_100s_gen_aug_run1 -gpu 0 -seed 0

python run.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/160.txt ./experiments/fastmri_recon_gen_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_160s_gen_aug_run1 -g multi 0.75 -g nraw 20 -gpu 0
python evaluate_recon.py -p ./experiments/fastmri_data_recon_gen.txt ./experiments/160.txt ./experiments/fastmri_recon_gen_aug.txt ./experiments/fastmri_varnet.txt -o ./results/varnet_160s_gen_aug_run1 -gpu 0 -seed 0

