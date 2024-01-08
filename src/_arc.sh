# install conda
# source miniconda/bin/activate
# cd arc/src
# conda activate arc
# smux new-session --partition=short --mem=10240 --time=00:20:00
# smux new-session --partition=m3i --mem=10240 --time=00:20:00
# smux new-session --partition=m3g --gres=gpu:1 --mem=10240 --time=00:20:00

module load cuda

sbatch _arc.job sst2 gpt2
sbatch _arc.job sst2 gpt2-m
sbatch _arc.job sst2 gpt2-l
sbatch _arc.job sst2 gpt2-xl
## sbatch _arc.job sst2 pythia-s
## sbatch _arc.job sst2 pythia-m
## sbatch _arc.job sst2 pythia-l
## sbatch _arc.job sst2 pythia-xl

sbatch _arc.job subj gpt2
sbatch _arc.job subj gpt2-m
sbatch _arc.job subj gpt2-l
sbatch _arc.job subj gpt2-xl
## sbatch _arc.job subj pythia-s
## sbatch _arc.job subj pythia-m
## sbatch _arc.job subj pythia-l
## sbatch _arc.job subj pythia-xl

sbatch _arc.job agnews gpt2
sbatch _arc.job agnews gpt2-m
sbatch _arc.job agnews gpt2-l
sbatch _arc.job agnews gpt2-xl
## sbatch _arc.job agnews pythia-s
## sbatch _arc.job agnews pythia-m
## sbatch _arc.job agnews pythia-l
## sbatch _arc.job agnews pythia-xl

sbatch _arc.job cb gpt2
sbatch _arc.job cb gpt2-m
sbatch _arc.job cb gpt2-l
sbatch _arc.job cb gpt2-xl
## sbatch _arc.job cb pythia-s
## sbatch _arc.job cb pythia-m
## sbatch _arc.job cb pythia-l
## sbatch _arc.job cb pythia-xl

sbatch _arc.job cr gpt2
sbatch _arc.job cr gpt2-m
sbatch _arc.job cr gpt2-l
sbatch _arc.job cr gpt2-xl
## sbatch _arc.job cr pythia-s
## sbatch _arc.job cr pythia-m
## sbatch _arc.job cr pythia-l
## sbatch _arc.job cr pythia-xl

sbatch _arc.job dbpedia gpt2
sbatch _arc.job dbpedia gpt2-m
sbatch _arc.job dbpedia gpt2-l
sbatch _arc.job dbpedia gpt2-xl
## sbatch _arc.job dbpedia pythia-s
## sbatch _arc.job dbpedia pythia-m
## sbatch _arc.job dbpedia pythia-l
## sbatch _arc.job dbpedia pythia-xl

sbatch _arc.job mpqa gpt2
sbatch _arc.job mpqa gpt2-m
sbatch _arc.job mpqa gpt2-l
sbatch _arc.job mpqa gpt2-xl
## sbatch _arc.job mpqa pythia-s
## sbatch _arc.job mpqa pythia-m
## sbatch _arc.job mpqa pythia-l
## sbatch _arc.job mpqa pythia-xl

sbatch _arc.job mr gpt2
sbatch _arc.job mr gpt2-m
sbatch _arc.job mr gpt2-l
sbatch _arc.job mr gpt2-xl
## sbatch _arc.job mr pythia-s
## sbatch _arc.job mr pythia-m
## sbatch _arc.job mr pythia-l
## sbatch _arc.job mr pythia-xl

sbatch _arc.job rte gpt2
sbatch _arc.job rte gpt2-m
sbatch _arc.job rte gpt2-l
sbatch _arc.job rte gpt2-xl
## sbatch _arc.job rte pythia-s
## sbatch _arc.job rte pythia-m
## sbatch _arc.job rte pythia-l
## sbatch _arc.job rte pythia-xl

sbatch _arc.job trec gpt2
sbatch _arc.job trec gpt2-m
sbatch _arc.job trec gpt2-l
sbatch _arc.job trec gpt2-xl
## sbatch _arc.job trec pythia-s
## sbatch _arc.job trec pythia-m
## sbatch _arc.job trec pythia-l
## sbatch _arc.job trec pythia-xl



# show_cluster
# show_job
# smux attach-session -t <job-id>
# scancel <job-id>
# sbatch arc.job

# nvidia-smi
# jobs -l
# ps -f | grep nohup
# exit