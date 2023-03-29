torchrun --nproc_per_node=8 finetune.py \
        --base_model '../../model/llama-7b-hf' \
        --data_path '../../data/alpaca-cleaned' \
        --output_dir './lora-alpaca' \
        --batch_size 16 \
        --micro_batch_size 16 \
        --num_epochs 3 \
        --learning_rate 1e-4 \
        --cutoff_len 512 \
        --val_set_size 2000 \
        --lora_r 8 \
        --lora_alpha 16 \
        --lora_dropout 0.05 \
        --lora_target_modules '[q_proj,v_proj]' \
        --train_on_inputs \
        --group_by_length
