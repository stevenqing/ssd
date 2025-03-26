#!/bin/bash


python run_scripts/sb3_independent.py \
--model social_influence \
--env-name lbf10 \
--num-cpus 12 \
--num-envs 32 \
--num-agents 3 \
--seed 1 \
--extractor cbam \
--total-timesteps 100_000_000 \

