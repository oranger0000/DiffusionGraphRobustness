#!/bin/bash

# Sample usage $1-input file, $2-output folder $3- batch size -5
# bash nmt_transform.sh ../data/exp/all_amt_data_small.csv ../data/exp/ 5

echo "Sample usage: bash run_all.sh sgc cora 0.05"
echo "Base network options: chebnet, sgc, gat"
echo "Dataset options: cora, cora_ml, citeseer, pubmed, polblogs"
echo "Perturbration ratio options: 0.05, 0.1, 0.15, 0.2, 0.25"

base_network=$1
dataset=$2
ptbrate=$3

if [[ $base_network == 'chebnet' ]]; then
    python3 test_chebnet_diffusion.py --dataset=$dataset --ptb_rate=$ptbrate
elif [[ $base_network == 'sgc' ]]; then
    python3 test_sgc_diffusion.py --dataset=$dataset --ptb_rate=$ptbrate
elif [[ $base_network == 'gat' ]]; then
  python3 test_gat_diffusion.py --dataset=$dataset --ptb_rate=$ptbrate
else
  echo "Choose base network between chebnet, sgc, and gat."
fi



