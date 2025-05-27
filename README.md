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
