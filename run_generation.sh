# Train generation network: CSM
python run.py -p ./experiments/fastmri_data_csm_lr.txt ./experiments/20.txt ./experiments/AAE.txt -o ./results/csm_aae_20s -gpu 0

# Train generation network: Phase
python run.py -p ./experiments/fastmri_data_phase_lr.txt ./experiments/20.txt ./experiments/AAE.txt -o ./results/phase_aae_20s -gpu 0

# Train super resolution network: CSM
python run.py -p ./experiments/fastmri_data_csm.txt ./experiments/20.txt ./experiments/fastmri_superres_csm.txt -o ./results/csm_superres_20s -gpu 0

# Train super resolution network: Phase
python run.py -p ./experiments/fastmri_data_phase.txt ./experiments/20.txt ./experiments/fastmri_superres_phase.txt -o ./results/phase_superres_20s -gpu 0
