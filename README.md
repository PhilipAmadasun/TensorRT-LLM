# TensorRT-LLM on x86\_64 (A6000, SM86) and Jetson Orin (SM87)

**Build, serve, and profile (including VRAM usage with Nsight)**

This branch documents a cross-arch build that targets both a workstation (x86\_64 + A6000, **SM86**) and NVIDIA Jetson Orin AGX (**SM87**), plus how to serve and **profile** it end-to-end.

**NOTE:** The original intent was to cross-compile for NVIDIA edge targets (e.g., Jetson Orin, `sm_87`): convert checkpoints to the TensorRT-LLM layout on a workstation, `scp` them to the device, and then build the final `.engine` on-device for `sm_87`. In practice this workflow is not viable today. The public `sm_87` backend in TensorRT-LLM lacks complete operator/kernel coverage for several newer LLM architectures and features, so engine build/serialization on Orin fails for those models. Until `sm_87` support is broadened, either (a) use model/configs known to be supported on Orin, (b) disable unsupported plugins/features, or (c) target `sm_86` GPUs for deployment.


---

## 0) Prerequisites

* NVIDIA driver + CUDA 12.x on the host.
* Docker with GPU access (`--gpus all`).
* `git-lfs` installed (TensorRT-LLM stores large files with LFS).
* (Optional) Nsight Systems / Nsight Compute for profiling.
* (Optional) `huggingface-cli` to fetch models.

---

## 1) Clone & initialize submodules

```bash
apt-get update && apt-get -y install git git-lfs
git lfs install

git clone https://github.com/NVIDIA/TensorRT-LLM.git
cd TensorRT-LLM
git submodule update --init --recursive
git lfs pull
```

---

## 2) Build a Docker image for **SM86** and **SM87**

> The key is passing real arch codes for both targets.

```bash
make -C docker release_build CUDA_ARCHS="'87-real;86-real'"
```

This produces a release image that contains CUDA objects for both `sm_87` (Orin) and `sm_86` (A6000).

**(Optional) Prebuilt image tag (example):**

```
uyiosaamadasun/tensorrt_llm_x86_64_sm86:latest
```

> Use your own registry/tag if you publish an image.

---

## 3) Verify which SMs the build supports (inside the container)

```bash
cuobjdump --list-elf /app/tensorrt_llm/lib/libnvinfer_plugin_tensorrt_llm.so \
  | grep -o "sm_[0-9]\+" | sort -u
```

You should see at least `sm_86` and `sm_87`.

---

## 4) Download a model (Hugging Face)

```bash
huggingface-cli download <huggingface-repo> \
  --local-dir /data/models/tinyllama-gptq \
  --local-dir-use-symlinks False
```

---

## 5) Convert / Quantize

### 5a) Convert a LLaMA-family checkpoint to TensorRT-LLM format

```bash
python /app/tensorrt_llm/examples/models/core/llama/convert_checkpoint.py \
  --model_dir <model dir> \
  --quant_ckpt_path </data/models/...model.safetensors> \
  --output_dir <output dir> \
  --dtype float16 \
  --use_weight_only \
  --weight_only_precision int4_gptq \
  --group_size 128 \
  --per_group
```

### 5b) Quantize from scratch to AWQ (TensorRT-LLM native layout)

```bash
python examples/quantization/quantize.py \
  --model_dir <model dir> \
  --qformat int4_awq \
  --output_dir <output dir> \
  --dtype float16
```

> **Tip:** Keep the quant configuration (group size, per-group, etc.) consistent between conversion and build. Mismatches hurt accuracy and performance.

---

## 6) Build the TensorRT engine

Minimal, single-GPU example:

```bash
trtllm-build \
  --checkpoint_dir <checkpoint> \
  --output_dir     <engine> \
  --weight_streaming \
  --max_batch_size 1 \
  --max_seq_len    512 \
  --max_num_tokens 512 \
  --gpt_attention_plugin disable \
  --fast_build
```

### Important flags & interactions (KV cache, paging, streaming)

* `--weight_streaming`
  Splits weights into a separate `*_managed_weights.safetensors` and DMA-streams layers into VRAM at runtime. Lets you run models larger than VRAM.

* `--kv_cache_type paged` *(default for GPT models)*
  Stores K/V in page chunks (e.g., 16/32 tokens per page) with a page table. Saves memory and enables KV-reuse.

* `--paged_state`
  Page-aware allocator for per-sequence state (including KV pages). **If KV is paged, paged\_state should be ON** so the attention kernel gets valid page tables.

* `--remove_input_padding` *(usually ON when paged KV is ON)*
  Packs variable-length sequences to cut prefill work and memory.

> **Heads-up:** In some builds **`--weight_streaming` disables `paged_state`** because the same pool is reused for the streaming buffer. If that happens you’ll see logs like `Set paged_state to False`. With paged KV and `remove_input_padding` enabled, that can produce half-initialized `attention_params` and early assertions in the attention path.
> **Fix:** Either build **without** `--weight_streaming` for paged KV, or keep KV **continuous** if you must stream weights.

---

## 7) Serving

### 7a) FastAPI example server (simple `/generate` endpoint)

```bash
python examples/apps/fastapi_server.py tinyllama_engine_int4wo \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --port 8000 \
  --max_beam_width 1 --tp_size 1 --pp_size 1 --cp_size 1 \
  --kv_cache_free_gpu_memory_fraction 0.8
```

Test:

```bash
curl -X POST http://<host>:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
        "prompt":"### Human: Explain KV-cache in 2 lines\n### Assistant:",
        "max_tokens":1000,
        "temperature":0.2,
        "streaming":false
      }'
```

### 7b) `trtllm-serve` (OpenAI-compatible REST)

```bash
trtllm-serve serve tinyllama_engine_int4wo \
  --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --host 0.0.0.0 \
  --port 8000
```

OpenAI-style request:

```bash
curl http://<host>:8000/v1/chat/completions \
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

---

## 8) Profiling (headless) with Nsight — **including VRAM usage**

You can do this entirely from the terminal. Two complementary tools:

* **Nsight Systems (`nsys`)** → end-to-end timeline (CPU↔CUDA), CUDA APIs, cuBLAS/cuDNN, memcopies, and NVTX ranges.
* **Nsight Compute (`ncu`)** → deep dive on **single kernels** (FMHA/attention, GEMM, etc.): occupancy, bandwidth, tensor-core utilization.

### 8.1 End-to-end capture (nsys)

**Run the server under `nsys`** (capture gated by CUDA profiler API so you only record the request):

```bash
nsys profile -o /tmp/tllm_serve \
  -t cuda,nvtx,osrt,cublas,cudnn \
  --capture-range=cudaProfilerApi \
  --force-overwrite=true \
  --export=sqlite,qdrep \
  --sample=none \
  trtllm-serve serve tinyllama_engine_int4wo \
    --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --host 0.0.0.0 --port 8000
```

Then issue **one** HTTP request (the app typically calls `cudaProfilerStart/Stop` around the critical section—hence you’ll see “Capture range started/ended”).

**Summaries (headless)**

```bash
# Build the SQLite once (if not already created)
nsys export --sqlite /tmp/tllm_serve.sqlite /tmp/tllm_serve.nsys-rep

# CUDA API + kernels + mem ops in one go
nsys stats -f table -r cuda_api_gpu_sum /tmp/tllm_serve.nsys-rep

# MemOps by time & by size (good for VRAM analysis)
nsys stats -f table -r cuda_gpu_mem_time_sum /tmp/tllm_serve.nsys-rep
nsys stats -f table -r cuda_gpu_mem_size_sum /tmp/tllm_serve.nsys-rep

# NVTX range summary (map phases like prefill/decode/enqueue to time)
nsys stats -f table -r nvtx_sum /tmp/tllm_serve.nsys-rep

# Kernel summaries (name, time, count)
nsys stats -f table -r cuda_gpu_kern_sum /tmp/tllm_serve.nsys-rep
```

> **Interpreting VRAM from `nsys` (headless):**
>
> * `cuda_gpu_mem_*` reports show **bytes transferred** and **memcpy/memset durations** (device<->device/host), plus API calls (`cudaMalloc`, `cudaMallocAsync`, etc.) with sizes.
> * Allocations seen here approximate **what the runtime requested**, not necessarily **total peak FB memory** due to driver pools, fragments, or TensorRT internal arenas.
> * To get a **peak VRAM number** in a terminal, combine `nsys` with `nvidia-smi` sampling:

```bash
# While your request runs (in a separate shell):
nvidia-smi --query-compute-apps=pid,process_name,used_memory \
           --format=csv -l 1 | grep -E 'python|trtllm|fastapi'
# or a per-GPU timeline:
nvidia-smi dmon -s pucm -d 1
```

Correlate peaks with NVTX ranges in `nsys` (`prefill`, `decode`, etc.).

> **Tip:** TensorRT-LLM prints helpful memory lines at startup and on KV setup (e.g., “Allocated X GiB for paged KV cache”, tokens-per-block, page counts). Keep `INFO` logging on to get those numbers—they’re ground truth for **KV cache capacity**.

### 8.2 Kernel deep dive (ncu)

Target only the hot kernels (FMHA/attention + GEMM) and **one** iteration to keep the report slim:

```bash
ncu --target-processes all --set full \
    --profile-from-start off \
    --launch-skip 20 --launch-count 1 \
    -k 'regex:.*(gpt_attention|fmha|attention|gemm).*' \
    -o /tmp/tllm_ncu \
    trtllm-serve serve tinyllama_engine_int4wo \
      --tokenizer TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
      --host 0.0.0.0 --port 8000
```

After the run:

```bash
# Text summary (no GUI)
ncu --import /tmp/tllm_ncu.ncu-rep --page summary --csv
# Useful metrics (bandwidth & occupancy)
ncu --import /tmp/tllm_ncu.ncu-rep --metrics \
  sm__throughput.avg.pct_of_peak_sustained_active, \
  smsp__pipe_tensor_throughput.avg.pct_of_peak_sustained_active, \
  lts__throughput.avg.pct_of_peak_sustained_active, \
  dram__throughput.avg.pct_of_peak_sustained_active \
  --csv
```

This explains kernel-level bottlenecks (occupancy vs memory-bound) but **not** total VRAM. Use it together with `nsys`/`nvidia-smi`.

---

## 9) VRAM quick reference (what uses memory)

* **Model params**

  * If **no** `--weight_streaming`: weights live in the `.engine` and resident in VRAM.
  * With `--weight_streaming`: weights live on disk as `*_managed_weights.safetensors` and a **subset** is DMA-streamed per layer. Instant VRAM is lower; bandwidth pressure is higher.

* **Workspaces / arenas** (TensorRT internal).
  Allocations not directly visible as your app’s `cudaMalloc`. Nsight will show API activity, but peak FB memory is best seen via `nvidia-smi`.

* **Activations + temporary buffers** (prefill).
  Scales with batch and input length; reduced with `--remove_input_padding`.

* **KV cache** (decode).
  Roughly: `bytes ≈ 2 * n_layers * n_kv_heads * head_dim * tokens * sizeof(kv_dtype)` per sequence, plus paging overhead.
  TensorRT-LLM logs page size and “tokens per block” (e.g., 32) and total pages allocated.

* **CUDA graphs / compiled plans** may hold extra resident buffers.

---

## 10) Troubleshooting

* **Device busy / unavailable** when starting `trtllm-serve`

  * Kill previous servers using the GPU; check `nvidia-smi` for stale PIDs.
  * Ensure you’re not in **exclusive mode** (`nvidia-smi -i 0 -q | grep -i "Compute Mode"`).
  * If profiling: run **one profiler at a time** (don’t overlap `nsys` and `ncu` on the same process).

* **Engine inspection says “profiling verbosity too low”**
  Rebuild the engine with **detailed** profiling verbosity (use the build config/yaml option for TensorRT engine verbosity) to get richer diagnostics at load time.

---

## 11) Appendix — exact `nsys stats` report names (CLI)

Useful built-ins:

* `cuda_api_gpu_sum` — CUDA API + Kernels + MemOps summary
* `cuda_gpu_kern_sum` — GPU kernel summary
* `cuda_gpu_mem_time_sum` — MemOps by **time**
* `cuda_gpu_mem_size_sum` — MemOps by **size**
* `nvtx_sum` — NVTX range summary (great to map phases)
* `osrt_sum` — OS runtime summary (threads, CPU)

Example:

```bash
nsys stats -f table \
  -r cuda_api_gpu_sum \
  -r cuda_gpu_kern_sum \
  -r cuda_gpu_mem_time_sum \
  -r cuda_gpu_mem_size_sum \
  -r nvtx_sum \
  /tmp/tllm_serve.nsys-rep
```

---

## 12) Recap: recommended profiling recipe

1. **End-to-end** (`nsys`) with `--capture-range=cudaProfilerApi`, then:

   * `cuda_api_gpu_sum`, `cuda_gpu_mem_*`, `nvtx_sum`.
   * Parallel `nvidia-smi ... -l 1` to capture **peak VRAM**.
2. **Deep dive** (`ncu`) on **FMHA/attention + GEMM** kernels for occupancy/bandwidth.
3. If using paged KV + remove-padding, avoid the `weight_streaming` ↔ `paged_state` conflict (or accept continuous KV).

---

That’s it. If you want, I can also drop a tiny shell script that runs the server under `nsys`, fires a single request, and prints the **peak VRAM** plus the top NVTX ranges automatically.
