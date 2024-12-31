export NCCL_P2P_LEVEL=NVL
export NCCL_IB_GID_INDEX=3
export HF_DATASETS_OFFLINE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

benchmark=gsm8k
n_shot=0
prompt_format=llama
max_new_tokens=512
max_input_tokens=2048
micro_batch_size_per_gpu=1

MODEL=Llama-3.2-1B-Instruct
MODEL_NAME_OR_PATH=./model_space/${MODEL}

VERIFIER=${MODEL}-Verifier
VERIFIER_MODEL_NAME_OR_PATH=./checkpoints/${VERIFIER}

alpha=0.1
beta=0.25


for beta in 0.25
do

args="
    --benchmark ${benchmark} \
    --data_dir ./data/evaluation/${benchmark} \
    --n_shot ${n_shot} \
    --max_new_tokens ${max_new_tokens} \
    --max_input_tokens ${max_input_tokens} \
    --micro_batch_size_per_gpu ${micro_batch_size_per_gpu} \
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --num_workers 8 \
    --preprocessing_num_workers 64 \
    --keep_in_memory \
    --num_beams ${num_beams} \
"

if [ "$prompt_format" != "None" ]; then
    args="${args} --prompt_format ${prompt_format}"
fi

if [ "$VERIFIER" == "None" ]; then
    args="
    ${args} \
    --save_dir ./results/${benchmark}/${MODEL}/${n_shot}-${prompt_format}-"${prefix}"-${max_new_tokens}-${max_input_tokens}-${micro_batch_size_per_gpu}-${num_beams}
"
else
    args="
    ${args} \
    --verifier_model_name_or_path ${VERIFIER_MODEL_NAME_OR_PATH} \
    --alpha ${alpha} \
    --beta ${beta} \
    --save_dir ./results/${benchmark}/${MODEL}/${VERIFIER}/${n_shot}-${prompt_format}-"${prefix}"-${max_new_tokens}-${max_input_tokens}-${micro_batch_size_per_gpu}-${strategy}-${alpha}-${beta}-${top_k}-${num_beams}
"
fi

accelerate launch evaluation.py $args

done