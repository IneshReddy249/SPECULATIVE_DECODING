# Convert draft model (FP16)
python3 trtllm_scripts/examples/qwen/convert_checkpoint.py \
    --model_dir /workspace/hf_models/qwen2.5-1.5b-instruct \
    --output_dir /workspace/checkpoints/draft \
    --dtype float16 \
    --tp_size 1

# Convert target model (INT8 weight-only)
python3 trtllm_scripts/examples/qwen/convert_checkpoint.py \
    --model_dir /workspace/hf_models/qwen2.5-32b-instruct \
    --output_dir /workspace/checkpoints/target_32b \
    --dtype float16 \
    --use_weight_only \
    --weight_only_precision int8 \
    --tp_size 1

