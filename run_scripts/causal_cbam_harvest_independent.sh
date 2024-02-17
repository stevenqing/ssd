python run_scripts/sb3_independent.py \
--env-name harvest \
--num-cpus 64 \
--num-envs 64 \
--num-agents 5 \
--seed 1 \
--extractor cbam \
--total-timesteps 100_000_000 \