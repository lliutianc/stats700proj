#!/bin/bash

#SBATCH --job-name=700-stage1
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
python3 -u lda_trial.py --model $1 --task $2 --rep_times 3 --metric ll --max_iter 200
echo $'FINISH:'$1' - n'
#python3 -u lda_trial.py --model $1 --task k --rep_times 3 --metric $2
echo $'FINISH:'$1' - k'
#python3 -u lda_trial.py --model $1 --task nk --rep_times 3 --metric $2
echo $'FINISH:'$1' - nk'
