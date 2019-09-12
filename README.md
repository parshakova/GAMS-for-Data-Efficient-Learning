# Global Autoregressive Models (GAMs)
Global Autoregressive Models for Data-Efficient Sequence Learning (official implementation for CONLL paper)

1. Make true data:
   - Call `wfsa_n_z.py` to create D,V,T datasets with a particular motif and process
   - motif PFSA: `python wfsa_n_z.py -prob_0 0.5 -length 30 -motif 1011100111001  -data_target ./data/pfsa_30_1011100111001 -valid 2000 -test 5000 -train 20000`
   - motif-anti-motif PFSA: `python wfsa_m.py -prob_0 0.5 -length 30 -motif 10001011111000 -second_select_prob 0.1 -second_length 30 -second_prob_0 0.5 -second_anti_motif 10001011111000 -second_selector_bit_remove -data_target ./data/pfsa_30_10001011111000.10001011111000 -valid 2000 -test 5000 -train 20000`
2. Train **r -> P_\lambda -> distillation (cyclic or no) -> \pi_\theta**
   - call `cycle_r_plambda_pitheta.py` with needed flags
   - e.g., `python -u cycle_r_plambda_pitheta.py --n 30 --ds_size 5000 --motif 4 --feat '1001111' --train 'snis_r' --mtype 'm' --restore yes --distill_size 20000 --cyclic`
   - or using slurm: call `pa_slurm_cycle_r_lambda_pi`, e.g. `sbatch --array=0  pa_slurm_cycle_r_lambda_pi`
3. Analyze the performace with Jupyter Notebook:
   - `jupyter notebook` or remotely `jupyter notebook --ip='0.0.0.0' --no-browser --port 8889`
   - `plot_conll_f12.ipynb.ipynb`
