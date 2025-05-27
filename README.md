# Tensorrt_llm build for x86_64 with attempted cross compile for SM 86 and SM 87
This branch will talk about the cross compile build that was tried for an x86_64 workstation (A6000 GPUs) and the Nvidia Jetson orin AGX.

Find out what GPU architectures your tensorrt-llm build supports in the Docker container
```
cuobjdump --list-elf /app/tensorrt_llm/lib/libnvinfer_plugin_tensorrt_llm.so \
  | grep -o "sm_[0-9]\+" | sort -u
```

Download your target model from huggingface
```
huggingface-cli download <huggingface repo>         --local-dir /data/models/tinyllama-gptq --local-dir-use-symlinks False
```

llama conversion
```
python /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py   --model_dir <model dir>   --quant_ckpt_path </data/models/....model.safetensors>  --output_dir <output dir>   --dtype float16   --use_weight_only   --weight_only_precision int4_gptq   --group_size 128 --per_group
```

quantizing from scratch with the tensorrt-llm format
```
python examples/quantization/quantize.py --model_dir <model dir> --qformat int4_awq --output_dir <output dir> --dtype float16
```

Build
```
trtllm-build \
  --checkpoint_dir <checkpoint> \
  --output_dir     <engine> \
  --weight_streaming \
  --max_batch_size 1 --max_seq_len 512 --max_num_tokens 512 \
  --gpt_attention_plugin disable --fast_build
```

Using openai API or curl
```
trtllm-serve serve tinyllama_engine_int4wo \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000
```
```
curl http://192.168.30.5:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
        "model": "TinyLlama-1.1B-Chat-v1.0",
        "messages": [
          {"role":"system","content":"You are a helpful assistant."},
          {"role":"user","content":"Where is New York?"}
        ],
        "max_tokens": 20,
        "temperature": 0.2
      }'
```

## Important parameters/flags for trtllm-build
Some parameters cannot be enabled at the same time for  a build as they would cause conflicts that lead to things like only partially innitiatlized attention parameter blocks.

```

1. --weight_streaming

    What it does – strips the weights out of the .engine and stores them in a
    separate *_managed_weights.safetensors file.
    At runtime the executor DMA-streams only the layers it needs into GPU memory, so even models that don’t fit in VRAM can run.
    NVIDIA GitHub

    Side effect – the builder forces paged_state = disable because
    paged-state memory is repurposed for the streaming weight buffer.
    (This isn’t obvious on the CLI; you just see Set paged_state to False in the log.)

2. --kv_cache_type paged (default for GPT)

    What it does – stores the key/value tensors in 16-token pages and keeps a
    page table. When the prompt is long the runtime can reuse or evict pages
    instead of allocating one gigantic tensor.
    It cuts KV-cache memory roughly in half and enables “KV-cache reuse” tricks.
    NVIDIA Developer

    Implicit default – when you build a decoder-only (GPT-style) model without
    passing --kv_cache_type, the builder sets paged_kv_cache = True →
    kv_cache_type = paged.

3. --paged_state

    What it is – a companion allocator that pages all per-sequence state
    (including the KV pages above) so they live in the same LRU pool.

    Why it must match – if the KV cache uses pages, the state allocator must be
    page-aware; otherwise the attention layer can’t find the page-table pointers
    it needs.
    Hence the guard in the kernel:

assert(attention_params && attention_params->is_valid());

4. --remove_input_padding (ON by default when paged KV is on)

    What it does – packs variable-length sequences back-to-back so the fused/
    flash-attention kernel (“FMHA”) only touches real tokens.
    Gives 20–30 % speed-up and halves prefill memory.
    docs.djl.ai

    Coupling – packing logic also relies on the same attention_params
    block, so if that block is half-initialised the FMHA path asserts early.

How the conflict happened
Flag	Value after parsing	Why
--weight_streaming	on	you asked for it
→ builder sets	paged_state = off	required for streaming buffer
(implicit)	kv_cache_type = paged	default for GPT
(implicit)	remove_input_padding = on	because cache is paged
Attention kernel checks	attention_params.is_valid()	fails (no page table)
Result: AssertionError during the dry-run inference pass inside the
builder.

```


```
                    +---------------------+
                    |  kv_cache_type      |
                    |   continuous / paged|
                    +----+----------+-----+
                         |          |
         needs page table|          | uses single tensor
                         |          |
                 +-------v---------+|
                 |  paged_state    ||  (any allocator OK)
                 |  enable/disable ||<---------------------+
                 +-------+---------+                      |
                         |                                |
          supplies page   |  supplies valid               |
            addresses      |  attention_params            |
                         v                                |
       +--------------------+          needs packed inputs|
       | remove_input_padding|----------------------------+
       |      ON / OFF       |          enables
       +----------+----------+          |
                  |                     |
                  v                     v
            +-------------------------------+
            |      FMHA / GPT-attention     |
            +-------------------------------+
```
