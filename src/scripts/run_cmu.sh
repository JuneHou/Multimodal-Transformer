# Run Python script with parameters
CUDA_LAUNCH_BLOCKING=1

python -W ignore ./src/scripts/main_cmu.py \
    --num_train_epochs 8 \
    --kernel_size 1 \
    --train_batch_size 2 \
    --eval_batch_size 8 \
    --seed 42 \
    --gradient_accumulation_steps 16 \
    --learning_rate 0.00002 \
    --layers 3 \
    --num_heads 8 \
    --output_dir "/home/jun/Workspace/results/" \
    --embed_dim 128 \
    --task 'multiclass' \
    --file_path '/home/jun/Workspace/datas/CMU-MOSI/' \
    --cross_method "moe" \
    --num_of_experts 16 \
    --top_k 2 \
    --router_type 'permod' \
    --hidden_size 512 \
    --gating_function "laplace"\
    --device 'cuda:4' \