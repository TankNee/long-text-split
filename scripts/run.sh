
#!/bin/bash

while getopts "n:g:d:" opt
do
    case "${opt}" in
        n )
            num_machines=${OPTARG}
            ;;
        g )
            gpus_per_node=${OPTARG}
            ;;
        d )
            bert_debug=${OPTARG}
            ;;
    esac
done
num_processes=$[num_machines * gpus_per_node]

if [ -z ${bert_debug} ]; then
    bert_debug=False
fi


# 读取分布式环境参数
if [ ${num_machines} -eq 1 ]; then
    # 单机设置
    MASTER_ADDR=localhost
    MASTER_PORT=6001
    RANK=0
else
    # 多机分布式设置
    #如果设备的编号为 0，则将其 ip 重置为数字 ip，方便 worker 读取数字 ip
    if [ ${RANK} -eq 0 ]; then
        MASTER_ADDR=$(hostname -i) export MASTER_ADDR=$MASTER_ADDR
    fi
fi
i=${RANK}

MODEL_NAME="/workspace/models/bert-base-chinese"
DATA_PATH=/workspace/coderepo/long-text-split/data/text/train_v3.jsonl
EVAL_DATA_PATH=/workspace/coderepo/long-text-split/data/text/test_v3.jsonl

torchrun --nnodes ${num_machines} --nproc_per_node ${gpus_per_node} --node_rank ${RANK} --master_port ${MASTER_PORT} --master_addr ${MASTER_ADDR} longtext/main.py \
  --model_name_or_path $MODEL_NAME \
  --data_path "$DATA_PATH" \
  --eval_data_path "$EVAL_DATA_PATH" \
  --output_dir output/longtext-v7 \
  --num_train_epoch 2 \
  --per_device_train_batch_size 2 \
  --per_device_eval_batch_size 16 \
  --gradient_accumulation_steps 8 \
  --eval_strategy "epoch" \
  --save_strategy "no" \
  --save_steps 200 \
  --save_total_limit 1 \
  --learning_rate 3e-4 \
  --weight_decay 0.1 \
  --adam_beta2 0.98 \
  --warmup_ratio 0.01 \
  --lr_scheduler_type "cosine" \
  --logging_steps 5 \
  --report_to "none" \
  --seed 3407 \
  --do_train \
  --do_eval \
  --do_distil \
  --bf16 True \
  --bert_debug ${bert_debug}

