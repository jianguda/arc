## install conda
# source miniconda/bin/activate
# cd arc/src
# conda activate arc
# smux new-session --partition=short --mem=10240 --time=00:20:00
# smux new-session --partition=m3i --mem=10240 --time=00:20:00
# smux new-session --partition=m3g --gres=gpu:1 --mem=10240 --time=00:20:00

module load cuda
time python3 arc.py
#nohup time python3 arc.py &

# show_cluster
# show_job
# smux attach-session -t <job-id>
# scancel <job-id>
# sbatch arc.job

# nvidia-smi
# jobs -l
# ps -f | grep nohup
# exit