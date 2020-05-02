#!/bin/bash
#SBATCH --job-name=cpc_model_test1
# Get email notification when job finishes or fails
#SBATCH --mail-type=END,FAIL # notifications for job done & fail
#SBATCH --mail-user=yuanfy@stanford.edu
# Define how long you job will run d-hh:mm:ss
#SBATCH --time 8:00:00
# GPU jobs require you to specify partition
#SBATCH --partition=V100
#SBATCH --nodes=1
#SBATCH --gres=gpu:1 
#SBATCH --mem=16G
# Number of tasks
#SBATCH --ntasks=1 
#SBATCH --cpus-per-task=8
