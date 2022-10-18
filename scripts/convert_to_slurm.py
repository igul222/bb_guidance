import os


SBATCH_PREFACE = \
"""#!/bin/bash
#SBATCH --partition=atlas --qos=normal
#SBATCH --time=7-00:00:00
#SBATCH --mem=60G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --exclude=atlas6,atlas7,atlas8,atlas9,atlas10,atlas16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1\
#SBATCH --job-name="{}.sh"
#SBATCH --error="{}/{}_err.log"
#SBATCH --output="{}/{}_out.log"\n
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
"""

counter = 0
OUTPUT_PATH="/atlas2/u/kechoi/bb_guidance/slurm/"

exp_id = 'noisy_scale10_fid_final'
# exp_id = 'baseline_scale10_fid'
script_fn = os.path.join(OUTPUT_PATH, '{}.sh'.format(exp_id))
counter += 1

base_cmd = 'python guidance.py --dataset cifar10 --classifier_type noisy --classifier_scale 10.0 --n_timesteps=1000 --n_samples=50000 --compute_fid=True --save_intermediate=False --exp_name=noisy_scale10_final'

# base_cmd = 'python guidance.py --dataset cifar10 --classifier_type baseline --classifier_scale 10.0 --n_timesteps=1000 --n_samples=50000 --compute_fid=True --save_intermediate=False --exp_name=baseline_scale10'

# write to file
with open(script_fn, 'w') as f:
    print(SBATCH_PREFACE.format(exp_id, OUTPUT_PATH, exp_id, OUTPUT_PATH, exp_id), file=f)
    print(base_cmd, file=f)
    print('sleep 1', file=f)
print('Generated {} experiment files'.format(counter))
#SBATCH --nodes=1
