# Draft engine
trtllm-build \
    --checkpoint_dir checkpoints/draft \
    --output_dir engines/draft \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_seq_len 2048

# Target baseline engine
trtllm-build \
    --checkpoint_dir checkpoints/target_32b \
    --output_dir engines/target_baseline \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_seq_len 2048

# Target speculative engine
trtllm-build \
    --checkpoint_dir checkpoints/target_32b \
    --output_dir engines/target_speculative \
    --gemm_plugin float16 \
    --gpt_attention_plugin float16 \
    --max_batch_size 1 \
    --max_input_len 1024 \
    --max_seq_len 2048 \
    --speculative_decoding_mode draft_tokens_external \
    --max_draft_len 5 \
    --use_paged_context_fmha enable

