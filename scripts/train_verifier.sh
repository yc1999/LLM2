export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_SL=3
export NCCL_NET_GDR_READ=1
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64
export LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

MODEL=Llama-3.2-1B-Instruct
MODEL_NAME_OR_PATH=./model_space/${MODEL}

OUTPUT_DIR=./checkpoints/${MODEL}-Verifier
mkdir -p $OUTPUT_DIR

NUM_PROCESSES=8
TOTAL_BATCH_SIZE=128
BATCH_SIZE_PER_GPU=16
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_PROCESSES/$BATCH_SIZE_PER_GPU))

ACCELERATE_LOG_LEVEL=info accelerate launch \
        --main_process_port 39999 \
        --num_processes $NUM_PROCESSES \
        --config_file config_files/accelerate_ds_zero3.yaml \
        train_verifier.py \
        config_files/verifier.yaml \
        --model_name_or_path=$MODEL_NAME_OR_PATH \
        --per_device_train_batch_size=$BATCH_SIZE_PER_GPU \
        --per_device_eval_batch_size=$BATCH_SIZE_PER_GPU \
        --gradient_accumulation_steps=$GRADIENT_ACC_STEPS  > $OUTPUT_DIR/output.txt 2>&1