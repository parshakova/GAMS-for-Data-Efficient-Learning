#!/bin/bash
#SBATCH -J tjarray
#SBATCH --gres=gpu:1
#SBATCH --output=/tmp-network/user/tparshak/tuned_%A_%a.log
#SBATCH -t 48:00:00
#SBATCH --mem=30000
#SBATCH --cpus-per-task=4
#SBATCH -p papago
#SBATCH -A papago
hostname
srun hostname
srun nvidia-smi -L
cd /tmp-network/user/tparshak/pg_methods

seqlen=( 10 )
motif=( 1 )
dssize=( 50  )
feats=( '1000000' )
train_reg=( 'snis_r' 'snis_mix' )
mtypes=( 'm' )
val_distill_size=10000
train2_reg='wn_dpg'
dgopt='normal'
v_rl_lr=0.001295
v_rl_mini_batch=128
v_rl_scale_iter=100
v_rl_patience=8
v_epochs=50

for n in ${seqlen[@]}; do
	for m in ${motif[@]}; do
		for ds in ${dssize[@]}; do
			for f in ${feats[@]}; do
				for tr in ${train_reg[@]}; do
					for mt in ${mtypes[@]}; do
						echo "$ds||$m||$n||$f||$tr||$mt"
						/home/tparshak/anaconda3/envs/py36/bin/python -u r_plambda_pitheta_full.py --n ${n} --ds_size ${ds} --motif ${m} --job ${SLURM_JOB_ID} --feat ${f} --train ${tr} --mtype ${mt} --restore yes --distill_size ${val_distill_size} --train2 ${train2_reg} --optim adam --debug_opt ${dgopt}  --wandb --rl_lr ${v_rl_lr} --rl_mini_batch ${v_rl_mini_batch} --rl_scale_iter ${v_rl_scale_iter} --rl_patience ${v_rl_patience} --epochs ${v_epochs}
						echo "$ds||$m||$n||$f||$tr||$mt" 
					done
				done
			done
		done
	done
done
scontrol show job ${SLURM_ARRAY_JOB_ID}
echo "SLURM_JOBID=$SLURM_JOBID"
echo "SLURM_ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID"
echo "SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
cp /tmp-network/user/tparshak/tuned_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log /home/tparshak/logs/tuned_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
rm -f /tmp-network/user/tparshak/tuned_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log
