#!/bin/bash

#SBATCH --job-name=700-stage2
#SBATCH --mail-user=liutianc@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32000m 
#SBATCH --time=10:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/outputs/%x-%j.log

# <include the code as if you are running it from the terminal>
python3 -u lda_svm.py --model $1 --task $2
echo $'FINISH:'$1'