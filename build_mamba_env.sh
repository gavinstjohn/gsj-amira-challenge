#~/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate base

mamba remove --name gsj-ac --all

mamba env create -f ./conda/gsj-ac.yaml