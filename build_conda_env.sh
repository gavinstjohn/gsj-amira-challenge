#~/usr/bin/env bash

eval "$(conda shell.bash hook)"
conda activate base

conda remove --name gsj-ac --all

conda env create -f ./conda/gsj-ac.yaml