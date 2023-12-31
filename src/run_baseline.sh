export CUDA_VISIBLE_DEVICES=0

LLM=gpt2-xl
LLM_DIR=./llm/${LLM}
DATA_DIR=./data/

# Set max demonstration shot w.r.t. context length
# max context length = 1024
array1=(mpqa) # maxshot = 32
array2=(sst2) # maxshot = 16
array3=(subj cr mr trec) # maxshot = 8
array4=(rte) # maxshot = 4
array5=(agnews cb) # maxshot = 2
array6=(dbpedia) # maxshot = 1

# for DATASET in sst2 subj mpqa agnews cb cr dbpedia mr rte trec; do
DATASET=sst2

if [[ "${array1[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=32
elif [[ "${array2[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=16
elif [[ "${array3[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=8
elif [[ "${array4[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=4
elif [[ "${array5[@]}" =~ "${DATASET}" ]]; then
N_DEMO_SHOT=2
else
N_DEMO_SHOT=1
fi

python3 icl.py \
    --llm_dir ${LLM_DIR} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --n_train_shot ${N_DEMO_SHOT} \
    --output_dir ./output

N_TRAIN_SHOT=1024
for KNN in 1 2 4 8 16 32 64 128 256 512 1024; do

python3 knn_prompting.py \
    --llm_dir ${LLM_DIR} \
    --dataset ${DATASET} \
    --data_dir ${DATA_DIR} \
    --n_train_shot ${N_TRAIN_SHOT} \
    --n_demo_shot ${N_DEMO_SHOT} \
    --output_dir ./output \
    --knn ${KNN}

done
