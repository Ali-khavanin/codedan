# Baseline Run Summary: `qwen3-8b-base-fp8_v1`

- **Run ID**: `qwen3-8b-base-fp8_v1`
- **Model**: `Qwen/Qwen3-8B-Base` (`49e3418fbbbca6ecbdf9608b4d22e5a407081db4`)
- **Precision**: FP8 weights & activations (`quantization="fp8"`, dynamic scales); `bfloat16` dtype & KV cache
- **Hardware**: `NVIDIA GeForce RTX 4090` (48 GiB / 49,140 MiB VRAM)
- **Software**: `vllm 0.32.0`, Python `3.12.3`
- **Run Window (UTC)**: `2026-11-26 06:55:53` – `09:34:29` (~2h 39m total: ~2h 21m generation + ~14m grading)
- **Integrity**: `14/14` files verified against `qwen3-8b-base-fp8_v1/SHA256SUMS`; `compare_runs.py` self-comparison passed with zero deltas.

## Benchmark Results

| Benchmark | Metric | Primary Score (`<answer>`) | Fallback Score (incl. fenced code) | GRP Paper Ref (full set) | Format Compliance | Syntax Error Rate | Truncation Rate | Mean Reasoning Tokens | Mean Total Tokens | Tokens / Correct | Gen Wall Clock | Throughput |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **MBPP** | `avg@16` | **21.69%** (`0.2169`) | **31.00%** (`0.3100`) | 73.27% | 0.0% | 5.97% | 29.38% | 2,297.5 | 2,603.5 | 12,004.5 | 4,074.2s (67.9m) | 1,022.4 tok/s |
| **MBPP+** | `avg@16` | **27.19%** (`0.2719`) | **37.62%** (`0.3762`) | 61.73% | 0.0% | 7.02% | 29.38% | 2,277.0 | 2,600.8 | 9,566.3 | 4,072.4s (67.9m) | 1,021.8 tok/s |
| **LiveCodeBench v5** | `avg@3` | **15.00%** (`0.1500`) | **26.33%** (`0.2633`) | 29.76% | 0.0% | 3.87% | 6.67% | 751.9 | 1,028.2 | 6,854.9 | 336.8s (5.6m) | 915.9 tok/s |

### LiveCodeBench v5 (`avg@3` by difficulty)
- **Easy**: `33.33%` (`0.3333`)
- **Medium**: `11.40%` (`0.1140`)
- **Hard**: `0.00%` (`0.0000`)

## Saved Artifacts
- `baseline/results/qwen3-8b-base-fp8_v1/` — Frozen run directory (untouched; matches `SHA256SUMS` 14/14):
  - `manifest.json`, `run.log`, `SHA256SUMS`
  - `mbpp/`, `mbpp_plus/`, `livecodebench_v5/` (`raw_generations.jsonl`, `eval_results.jsonl`, `summary.json`, `generation_wall_clock_sec.txt`)
- `baseline/results/qwen3-8b-base-fp8_v1_executed.ipynb` — Executed notebook with rendered cell outputs
- `baseline/results/qwen3-8b-base-fp8_v1_nbconvert.log` — Full `nbconvert` execution log
- `datasets/revisions.json` — Pinned dataset commits
- `datasets/samples/` — Fixed 100-task sample IDs reused by all subsequent thesis experiments (`mbpp_task_ids.json`, `mbpp_plus_task_ids.json`, `livecodebench_v5_task_ids.json`)
