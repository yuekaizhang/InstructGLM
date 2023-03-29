
MODEL=../../model/llama-7b-hf
FSDP_LAYER=LlamaDecoderLayer


DATA=../../data/alpaca_cleaned/alpaca_data_cleaned.json

EXE=./train.py
GPUS=8
SAVE_STR=epoch  # no, steps, epochs
SAVE_STEPS=1
BATCH=2
ACCUM=1   # Batch = BATCH * ACCUM * GPU
LR=1e-5
MAX_SEQ_LENGTH=1024
EPOCHS=3

DATANAME=alpaca
OUTPUT=./exp/llama-alpaca${MAX_SEQ_LENGTH}
mkdir -p ${OUTPUT}

# -d means detach, so docker runs on the back
# --max_steps 256 \
TRANSFORMERS_VERBOSITY=info \
torchrun --nproc_per_node=${GPUS} --master_port=5999 ${EXE} \
  --num_train_epochs ${EPOCHS} \
  --save_strategy ${SAVE_STR} \
  --save_steps ${SAVE_STEPS} \
  --save_total_limit 3 \
  --model_name_or_path ${MODEL} \
  --data_path ${DATA} \
  --dataset_name ${DATANAME} \
  --output_dir ${OUTPUT} \
  --fp16 \
  --model_max_length ${MAX_SEQ_LENGTH} \
  --per_device_train_batch_size ${BATCH} \
  --gradient_accumulation_steps ${ACCUM} \
  --evaluation_strategy no \
  --learning_rate ${LR} \
  --weight_decay 0. \
  --warmup_ratio 0.03 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --fsdp 'full_shard auto_wrap' \
  --fsdp_transformer_layer_cls_to_wrap '${FSDP_LAYER}'
