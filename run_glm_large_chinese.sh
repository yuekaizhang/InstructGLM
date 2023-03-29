jsonl_name=data/belle_1M_data.jsonl
data_path=data/belle_1M_dataset
dataset_name=BelleGroup/generated_train_1M_CN

# max_seq_length=512

# dataset_name=yahma/alpaca-cleaned
# jsonl_name=data/alpaca_cleaned.jsonl
# data_path=data/alpaca_cleaned_dataset
# max_seq_length=750

# mkdir -p $data_path
# python3 covert_dataset2jsonl.py \
#     --dataset_name $dataset_name \
#     --save_path $jsonl_name

# python3 tokenize_dataset_rows.py \
#     --jsonl_path $jsonl_name \
#     --save_path $data_path \
#     --max_seq_length $max_seq_length # for input or output, rather than input+output

# git-lfs install
# git clone https://huggingface.co/datasets/yuekai/belle_1M_and_alpaca_cleaned.git data


model_path=/mnt/samsung-t7/yuekai/llm/models/glm-large-chinese
data_path=data/belle_100k_dataset_glm_large_chinese
torchrun --nproc_per_node=1 finetune_glm_large_chinese.py \
    --dataset_path $data_path \
    --lora_rank 8 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 1 \
    --model_path $model_path \
    --save_steps 1000 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --fp16 \
    --remove_unused_columns false \
    --logging_steps 50 \
    --output_dir output
