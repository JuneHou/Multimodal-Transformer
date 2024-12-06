export CUDA_VISIBLE_DEVICES="4" 

python -W ignore ./src/scripts/main_mimiciv.py  --num_train_epochs 8  --modeltype 'CXR' \
                --kernel_size 1 --train_batch_size 2 --eval_batch_size 8 --seed 42 \
                --gradient_accumulation_steps 16  --num_update_bert_epochs 2 --bertcount 3 \
                --ts_learning_rate 0.0004 --txt_learning_rate 0.00002 \
                --notes_order 'Last' --num_of_notes 5 --max_length 1024 --layers 3\
                --output_dir "/data/wang/junh/results/Fuse_moe/all_los/" \
                --embed_dim 128 \
                --num_modalities 1 \
                --model_name "bioLongformer"\
                --task 'ihm-48-cxr-notes-ecg'\
                --file_path '/data/wang/junh/datasets/multimodal/preprocessing'\
                --num_labels 2 \
                --num_heads 8\
                --embed_time 64\
                --tt_max 48\
                --TS_mixup\
                --mixup_level 'batch'\
                --fp16 \
                --irregular_learn_emb_cxr \
                --cross_method "moe" \
                --gating_function "laplace" "laplace" \
                --num_of_experts 16 5 \
                --top_k 2 4 \
                --disjoint_top_k 2 \
                --hidden_size 512 \
                --use_pt_text_embeddings \
                --router_type 'permod' \