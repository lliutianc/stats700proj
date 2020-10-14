#!/bin/bash

#SBATCH --job-name=700-stage1
#SBATCH --mail-user=liutianc@umich.edu
#SBATCH --mail-type=BEGIN,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=32000m 
#SBATCH --time=10:00:00
#SBATCH --account=stats_dept1
#SBATCH --partition=standard
#SBATCH --output=/home/%u/outputs/%x-%j.log

# <include the code as if you are running it from the terminal>
python3 -u lda_trial.py --model lda --task n --rep_times 5 --metric pp --max_iter 500 
echo $'FINISH: LDA - n'
python3 -u lda_trial.py --model lda --task k --rep_times 5 --metric pp --max_iter 500
echo $'FINISH: LDA - k'
python3 -u lda_trial.py --model lda --task nk --rep_times 5 --metric pp --max_iter 500
echo $'FINISH: LDA - nk'

python3 -u lda_trial.py --model ctm --task n --rep_times 5 --metric pp --max_iter 500 
echo $'FINISH: CTM - n'
python3 -u lda_trial.py --model ctm --task k --rep_times 5 --metric pp --max_iter 500
echo $'FINISH: CTM - k'
python3 -u lda_trial.py --model ctm --task nk --rep_times 5 --metric pp --max_iter 500
echo $'FINISH: CTM - nk'
