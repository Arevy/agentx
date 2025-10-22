# CLI Programming Agent

Local CLI-controlled agent built on top of open-source Hugging Face models. It reads local documentation, uses persistent memory, and can execute safe commands within the workspace. The agent now supports large-context instruction models (Qwen, Devstral) with optional quantisation, JSON tool-calling, and persistent audit logs.

## Model & Hardware Strategy

| Model | Params / Context | Indicative Scores* | Deployment Notes |
|-------|------------------|--------------------|------------------|
| **Devstral-Small 1.0** | ~8 B dense · 128 K ctx | SWE-Bench Verified ≈ 25, HumanEval ≈ 63 | Fits today’s RTX 2070 Super (8 GB) only with 4-bit quantisation (`--quantization 4bit`) and light CPU offload. Recommended daily driver until hardware upgrade. |
| **Qwen2.5-Coder-32B** | 32 B dense · 128 K ctx | SWE-Bench Verified ≈ 38, HumanEval ≈ 82 | Needs ≥ 2×24 GB GPUs or aggressive 4-bit loading on a 64 GB RAM host. Use once more compute arrives. |
| **Qwen3-Coder-480B-A35B-Instruct** | 480 B MoE · 256 K ctx | SWE-Bench Verified ≈ 52, HumanEval ≈ 92 | Requires multi-GPU (≥ 8×80 GB) or vLLM + CPU offload with ≥ 500 GB RAM. Target for the future dual-Xeon cluster with MoE offloading. |

\*Scores compiled from vendor model cards / community leaderboards. Use them for relative comparison only.

- **Current machine (RTX 2070S, 64 GB RAM):** run Devstral-Small (or Qwen2.5-Coder-7B as fallback) in 4-bit via `--quantization 4bit --device-map auto --torch-dtype float16`.
- **Future workstation (dual Xeon, 500 GB RAM, high-end GPUs):** deploy Qwen3-Coder-480B using vLLM with MoE offloading; Qwen2.5-Coder-32B acts as intermediate step.
- **Why Devstral now?** Balanced instruction-following on coding tasks, native tool-calling, and manageable VRAM footprint when quantised.

## 1. Environment Setup
- Requires Python 3.10+ and optionally an NVIDIA GPU with a valid driver/CUDA stack.
- Create the virtual environment:
  ```bash
  cd agent
  python3 -m venv .venv
  source .venv/bin/activate
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
- Optional (Linux CUDA): `pip install bitsandbytes` to enable 4-bit/8-bit quantisation (`--quantization 4bit|8bit`). On macOS/Windows stay CPU-only or evaluate `auto-gptq`.
- If you are running on CPU only, keep the `batch-size` small (e.g., 1) and increase `gradient_accumulation_steps` to maintain a reasonable effective batch.
- The FAISS memory layer downloads `sentence-transformers/all-MiniLM-L6-v2` on first run; ensure the process has internet access or pre-cache the model in `~/.cache/huggingface`.
- For context7, set `CONTEXT7_URL` and/or use `--context7-url` so the agent can request additional documentation fragments when local sources fall short.

## 2. Data Collection
- Gather open-source code (e.g., The Stack, OSS GitHub projects), StackOverflow Q&A, and technical documentation.
- Normalize the text (UTF-8) and concatenate it into a large file with separated lines (`data/corpus_train.txt`).
- Set aside a small validation split (`data/corpus_eval.txt`) to monitor overfitting.

## 3. Fine-tuning
- Run `train.py` (GPU recommended for speed):
  ```bash
  source .venv/bin/activate
  python train.py \
      --model-name gpt2-medium \
      --train-data data/corpus_train.txt \
      --eval-data data/corpus_eval.txt \
      --output-dir models/gpt2-coder \
      --batch-size 1 \
      --gradient-accumulation-steps 32 \
      --epochs 3 \
      --fp16
  ```
- Key parameters:
  - `--block-size` – number of tokens per sample (1,024 for GPT-2 medium).
  - `--fp16/--bf16` – enable reduced precision on GPU; disable it on CPU.
  - `--learning-rate` – keep values low (5e-5 – 2e-5) for stability.
- After training, the model and tokenizer are saved in `models/gpt2-coder`.

## 4. Running the CLI Agent
- Start the agent with the fine-tuned model or any available open-source model:
  ```bash
  python agent_cli.py --model models/gpt2-coder --workspace /Users/rnd/Downloads/CODE
  ```
- Quick commands:
  - `--once "Describe the module main.py"` – single-shot response without the interactive loop.
  - `--remember "Useful note"` – immediately add an entry to memory.
- The agent executes exactly one tool per turn, deduplicates tool_calls, and logs the full flow (tool_calls + observations) in `logs/agent_audit.jsonl`. Both JSON (`tool_calls`) and legacy `Action:` formats are accepted; see `docs/local_agent_prompt.md` for correct/incorrect samples.
- Persistent memory relies on FAISS (`memory_bank/vector_store/`) with MiniLM embeddings; every message and note is indexed and automatically recalled in the prompt.

### Stronger local models
- **RTX 2070S target (quantised Devstral-Small):**
  ```bash
  python agent_cli.py \
      --model mistralai/Devstral-Small-2507 \
      --device-map auto \
      --quantization 4bit \
      --torch-dtype float16 \
      --trust-remote-code \
      --low-cpu-mem-usage \
      --context-window 131072 \
      --max-new-tokens 1024 \
      --max-actions-per-turn 1 \
      --observation-summary-chars 320 \
      --disable-system-prompt \
      --disable-thinking
  ```
- **Future 500 GB RAM / multi-GPU:** deploy Qwen3-Coder-480B locally using [vLLM](https://github.com/vllm-project/vllm) once the dual-Xeon cluster is ready: `pip install vllm`, start `python -m vllm.entrypoints.api_server --model Qwen/Qwen3-Coder-480B-A35B-Instruct --tensor-parallel-size 8 --max-model-len 256000`, then point the CLI agent to the on-prem endpoint when the HTTP backend adapter lands.
- For CPU-only experiments, prefer `--device-map cpu --torch-dtype float32 --quantization none` and limit `--max-new-tokens`.
- Place any locally fine-tuned checkpoint under `models/<name>` and point `--model` to that directory.

### Context7 (Dynamic Documentation)
- Configure `CONTEXT7_URL` and optionally `CONTEXT7_API_KEY` so the agent can request documentation fragments through the `call_context7` tool when FAISS/local docs do not provide enough answers. A default key (`ctx7sk-840f286d-465e-4343-aadb-b4453dbcbae3`) is bundled; override it via `--context7-api-key` or the environment variable if you need a different credential.
- Example: `CONTEXT7_URL=https://context7.local/v1 python agent_cli.py --model mistralai/Devstral-Small-2507 --context7-url https://context7.local/v1 ...`
- Retrieved passages are deduplicated against FAISS results and truncated to 256 chars before inclusion so the model only sees fresh information.
- Automatic fallback injects context7 matches into the prompt whenever local search yields nothing relevant and surfaces HTTP errors as observations when the service is down.

### Extra CLI knobs
- `--device-map` – explicit device placement (`auto`, `cuda`, `cpu`, `mps`).
- `--torch-dtype` – force precision (`float16`, `bfloat16`, `float32`).
- `--trust-remote-code` – required for models with custom generation code (Qwen family).
- `--low-cpu-mem-usage` – uses streaming weight loading to handle large models in limited RAM.
- `--quantization {none,8bit,4bit}` – wraps `BitsAndBytesConfig` for VRAM constrained devices.
- `--context-window` – override tokenizer window (up to 128K / 256K depending on model).
- `--max-actions-per-turn` – limits consecutive tool executions (default 1).
- `--observation-summary-chars` – summarises observations before reinjecting them into the prompt.
- `--audit-log` – sets the JSONL log file (defaults to `logs/agent_audit.jsonl`).
- `--system-prompt`, `--disable-system-prompt` – customise or remove initial instructions (Qwen3 non-thinking mode).
- `--enable-thinking/--disable-thinking` – passes `enable_thinking` to models that expose deliberate reasoning toggles.
- Newly introduced tools: `search_web`, `install_package`, `run_python`, `download_file`, `call_context7` (see `docs/local_agent_prompt.md` for valid JSON and selection rules). `run_python` enforces an expanded deny-list (`os`, `subprocess`, `builtins`, `sys`, `importlib`), `download_file` retries transient failures with a 5 MB cap, and `call_context7` only triggers when local context is empty.
- `--memory-vector-dir` – the FAISS directory used for persistent vector memory.
- `--context7-url`, `--context7-api-key` – configure the context7 server for external documentation (falls back when FAISS/local docs are insufficient).

## 5. Project Structure
```
agent/
├── agent_cli.py          # Main Reason→Act→Observe loop
├── train.py              # Fine-tuning script using Hugging Face Trainer
├── requirements.txt      # Core dependencies (transformers, datasets, torch, etc.)
├── README.md             # This document
├── docs/                 # Local documentation for simple RAG
├── memory_bank/global.md # Persistent memory with timestamps
└── models/               # Directory for fine-tuned models
```

## 6. Containerization
- Build the image:
  ```bash
  docker build -t myagent .
  ```
- Run locally on GPU (if available) and mount the current workspace:
  ```bash
  docker run -it --rm --gpus all \
      -v "$(pwd)":/workspace \
      -w /workspace/agent \
      myagent
  ```
- Remove the `--gpus all` flag when running on CPU.

## 7. Extensions and Scaling
- Larger models (require more VRAM):
  - **Devstral-Small 1.0** (8 B) – daily driver with 4-bit quantisation on single 8 GB GPU; CPU fallback works but is slower.
  - **Qwen2.5-Coder 14B/32B** – progressively higher quality; expect 4-bit quantisation on a 64 GB RAM workstation or ≥ 48 GB GPU VRAM.
  - **Qwen3-Coder-480B-A35B-Instruct** – state-of-the-art; run via vLLM with tensor + expert parallelism on the future multi-GPU rig.
- Framework integrations:
  - **LangChain**, **CrewAI**, **LangGraph** can manage tools, advanced memory, and multi-agent workflows.
- Advanced persistent memory:
  - Extend `memory_bank` with a FAISS vector store or integrate Memory Bank services (Vertex AI) for semantic search.
- Future work: plug a vLLM backend (`--model http://localhost:8000/v1`) and attach external RAG sources (StackOverflow, GitHub issues) to `search_web` once available.

## 8. Quick Troubleshooting
- If the model requests blocked commands, craft the prompt to explicitly explain they are forbidden.
- For GPU OOM issues, reduce `--batch-size` and increase `--gradient-accumulation-steps`.
- When using only CPU, disable `--fp16/--bf16`, set `--quantization none`, and consider lowering `max_new_tokens` and `context_window`.
- Consult `docs/local_agent_prompt.md` for prompt structure, valid tool_call examples, and common mistakes.
- Run the tool-selection test: `python tests/test_tool_selection.py` to confirm requests such as “list files” map to `run_command`.
