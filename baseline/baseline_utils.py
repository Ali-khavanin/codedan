"""Helpers for baseline.ipynb: datasets, sampling, prompts, generation, grading, metrics.

The notebook stays short by calling these functions. Every function is plain
Python with no hidden state, so results can be re-graded on CPU from the saved
raw generations at any time.
"""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import os
import pickle
import random
import re
import subprocess
import sys
import tempfile
import time
import zlib
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

DATASETS = {
    "mbpp": {
        "repo": "google-research-datasets/mbpp",
        "files": ["full/test-00000-of-00001.parquet"],
        "expected_tasks": 500,
    },
    "mbpp_plus": {
        "repo": "evalplus/mbppplus",
        "files": ["data/test-00000-of-00001-d5781c9c51e02795.parquet"],
        "expected_tasks": 378,
    },
    # release_v5 of LiveCodeBench = the first five JSONL shards.
    "livecodebench_v5": {
        "repo": "livecodebench/code_generation_lite",
        "files": [f"test{s}.jsonl" for s in ["", "2", "3", "4", "5"]],
        "expected_tasks": 880,
    },
}


def download_datasets(datasets_dir: Path, revisions: dict[str, str]) -> None:
    """Download every dataset file at its pinned commit into datasets/<name>/raw/."""
    from huggingface_hub import hf_hub_download

    for name, spec in DATASETS.items():
        for filename in spec["files"]:
            hf_hub_download(
                spec["repo"],
                filename,
                repo_type="dataset",
                revision=revisions[name],
                local_dir=datasets_dir / name / "raw",
            )
    (datasets_dir / "revisions.json").write_text(
        json.dumps({n: {"repo": DATASETS[n]["repo"], "revision": r} for n, r in revisions.items()}, indent=2) + "\n"
    )


def load_mbpp(datasets_dir: Path) -> list[dict]:
    df = pd.read_parquet(datasets_dir / "mbpp/raw" / DATASETS["mbpp"]["files"][0])
    return [
        {
            "task_id": f"mbpp/{r.task_id}",
            "execution_mode": "function",
            "difficulty": None,
            "question": r.text,
            "example_test": r.test_list[0],
            "test_code": "\n".join([r.test_setup_code, *r.test_list]),
        }
        for r in df.itertuples()
    ]


def load_mbpp_plus(datasets_dir: Path) -> list[dict]:
    df = pd.read_parquet(datasets_dir / "mbpp_plus/raw" / DATASETS["mbpp_plus"]["files"][0])
    return [
        {
            "task_id": f"mbpp_plus/{r.task_id}",
            "execution_mode": "function",
            "difficulty": None,
            "question": r.prompt,
            "example_test": r.test_list[0],
            "test_code": "\n".join([*r.test_imports, r.test]),
        }
        for r in df.itertuples()
    ]


def _decode_lcb_tests(raw: str) -> list[dict]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return json.loads(pickle.loads(zlib.decompress(base64.b64decode(raw.encode()))))


def load_livecodebench(datasets_dir: Path, keep_tests_for: set[str] | None = None) -> list[dict]:
    """Load LCB v5. Hidden tests are decoded only for task ids in `keep_tests_for`."""
    tasks = []
    for filename in DATASETS["livecodebench_v5"]["files"]:
        with open(datasets_dir / "livecodebench_v5/raw" / filename) as f:
            for line in f:
                r = json.loads(line)
                task = {
                    "task_id": r["question_id"],
                    "execution_mode": "function" if r["starter_code"].strip() else "stdin",
                    "difficulty": r["difficulty"],
                    "platform": r["platform"],
                    "contest_date": r["contest_date"],
                    "question": r["question_content"],
                    "starter_code": r["starter_code"],
                    "func_name": json.loads(r["metadata"] or "{}").get("func_name"),
                }
                if keep_tests_for is None or task["task_id"] in keep_tests_for:
                    task["tests"] = json.loads(r["public_test_cases"]) + _decode_lcb_tests(r["private_test_cases"])
                tasks.append(task)
    return tasks


# ---------------------------------------------------------------------------
# Representative sampling
# ---------------------------------------------------------------------------


def uniform_sample(task_ids: list[str], n: int, seed: int) -> list[str]:
    """Simple random sample without replacement (order-independent of input)."""
    return sorted(random.Random(seed).sample(sorted(task_ids), n))


def stratified_sample(tasks: list[dict], n: int, seed: int, keys: tuple[str, ...]) -> list[str]:
    """Proportional stratified sample (largest-remainder allocation per stratum)."""
    strata = defaultdict(list)
    for t in tasks:
        strata[tuple(t[k] for k in keys)].append(t["task_id"])
    total = len(tasks)
    quotas = {s: n * len(ids) / total for s, ids in strata.items()}
    alloc = {s: int(q) for s, q in quotas.items()}
    for s in sorted(quotas, key=lambda s: (quotas[s] - alloc[s], s), reverse=True)[: n - sum(alloc.values())]:
        alloc[s] += 1
    rng = random.Random(seed)
    picked = []
    for s in sorted(strata):
        picked += rng.sample(sorted(strata[s]), alloc[s])
    return sorted(picked)


def strata_table(tasks: list[dict], sample_ids: list[str], keys: tuple[str, ...]) -> pd.DataFrame:
    """Share of each stratum in the full dataset vs the sample."""
    chosen = set(sample_ids)
    full = Counter(tuple(t[k] for k in keys) for t in tasks)
    samp = Counter(tuple(t[k] for k in keys) for t in tasks if t["task_id"] in chosen)
    rows = [
        {**dict(zip(keys, s)), "full_%": 100 * full[s] / len(tasks), "sample_%": 100 * samp[s] / len(chosen)}
        for s in sorted(full)
    ]
    return pd.DataFrame(rows).round(1)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

FORMAT_RULES = """\
Answer format:
- Put your step-by-step reasoning inside <think>...</think>.
- Then put only the final, executable Python code inside <answer>...</answer>.
  No prose and no Markdown fences inside the answer tags.
- All imports and top-level def/class statements start at column 0. Include every
  standard-library and typing import you use.

Example of the required format (for a different problem):
<think>
The task asks for the sum of the squares of a list, so I square each element and add them up.
</think>
<answer>
def sum_of_squares(nums):
    return sum(x * x for x in nums)
</answer>"""

# The assistant turn starts with this text (R1-Zero-style prefill for base models).
# It is stored as the first characters of every raw_output.
RESPONSE_PREFIX = "<think>\n"

PROMPT_TEMPLATES = {
    "function_completion": """\
You are an expert Python programmer. Solve the following problem.

{question}

{signature_hint}

Requirements:
- Keep the exact function signature, class structure, and argument names.
- Do not mutate input arguments in place.

""" + FORMAT_RULES,
    "online_judge_stdin": """\
You are an expert Python programmer. Solve the following competitive programming problem.

{question}

Requirements:
- Write a complete standalone Python program.
- Read all input from sys.stdin and write the result to sys.stdout.
- Do not print prompts or any extra text.

""" + FORMAT_RULES,
}


def template_hashes() -> dict[str, str]:
    hashes = {name: hashlib.sha256(t.encode()).hexdigest() for name, t in PROMPT_TEMPLATES.items()}
    return {**hashes, "response_prefix": hashlib.sha256(RESPONSE_PREFIX.encode()).hexdigest()}


def build_prompt(task: dict) -> str:
    if task["execution_mode"] == "stdin":
        return PROMPT_TEMPLATES["online_judge_stdin"].format(question=task["question"])
    if "starter_code" in task:  # LiveCodeBench functional problem
        hint = f"Complete this starter code:\n{task['starter_code']}"
    else:  # MBPP / MBPP+: one visible assert fixes the function name and signature
        hint = f"Your code must pass this test:\n{task['example_test']}"
    return PROMPT_TEMPLATES["function_completion"].format(question=task["question"], signature_hint=hint)


# ---------------------------------------------------------------------------
# Generation (vLLM, offline batched)
# ---------------------------------------------------------------------------

THINK_RE = re.compile(r"<think>(.*?)</think>", re.S)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.S)


def generate(llm, tokenizer, tasks: list[dict], k: int, decoding: dict, max_tokens: int, log) -> tuple[list[dict], float]:
    """Generate k samples per task. Returns raw records and total wall-clock seconds.

    Each sample is its own request with seed = base_seed + sample_idx (the same
    scheme vLLM uses internally for n>1), so per-sample latency is measurable.
    """
    from vllm import SamplingParams

    prompts, params, keys = [], [], []
    for task in tasks:
        chat = tokenizer.apply_chat_template(
            [{"role": "user", "content": build_prompt(task)}], tokenize=False, add_generation_prompt=True
        ) + RESPONSE_PREFIX
        for i in range(k):
            prompts.append(chat)
            params.append(
                SamplingParams(
                    temperature=decoding["temperature"],
                    top_p=decoding["top_p"],
                    seed=decoding["seed"] + i,
                    max_tokens=max_tokens,
                    stop=["</answer>"],
                    include_stop_str_in_output=True,
                )
            )
            keys.append((task, i))

    log(f"generating {len(prompts)} completions ({len(tasks)} tasks x {k})")
    start = time.perf_counter()
    outputs = llm.generate(prompts, params, use_tqdm=True)
    wall = time.perf_counter() - start
    log(f"done in {wall:.0f}s")

    def count(text: str | None) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False)) if text else 0

    records = []
    for (task, i), out in zip(keys, outputs):
        comp = out.outputs[0]
        text = RESPONSE_PREFIX + comp.text
        think, answer = reasoning_text(text), ANSWER_RE.search(text)
        m = out.metrics
        latency = (m.last_token_ts - m.queued_ts) if m and m.last_token_ts and m.queued_ts else None
        records.append(
            {
                "task_id": task["task_id"],
                "difficulty": task["difficulty"],
                "execution_mode": task["execution_mode"],
                "sample_idx": i,
                "raw_output": text,
                "finish_reason": comp.finish_reason,
                "reasoning_tokens": count(think),
                "answer_tokens": count(answer and answer.group(1)),
                "total_tokens": len(comp.token_ids),
                "latency_sec": round(latency, 3) if latency is not None else None,
            }
        )
    return records, wall


def reasoning_text(raw: str) -> str:
    """Reasoning = text after the prefix up to </think>; else up to <answer>; else to the end.

    The base model often never closes </think>, so requiring it would report 0 reasoning tokens.
    """
    body = raw[len(RESPONSE_PREFIX):] if raw.startswith(RESPONSE_PREFIX) else raw
    for marker in ("</think>", "<answer>"):
        if marker in body:
            return body.split(marker, 1)[0]
    return body


def recount_reasoning_tokens(records: list[dict], tokenizer) -> list[dict]:
    """Recompute reasoning_tokens from saved raw_output with the reasoning_text rule."""
    return [{**r, "reasoning_tokens": len(tokenizer.encode(reasoning_text(r["raw_output"]), add_special_tokens=False))}
            for r in records]


# ---------------------------------------------------------------------------
# Code extraction and sandboxed execution
# ---------------------------------------------------------------------------

FENCE_RE = re.compile(r"^\s*```[a-zA-Z0-9]*\s*\n(.*?)\n\s*```\s*$", re.S)
LAST_PY_FENCE_RE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)\n[ \t]*```", re.S)


def extract_code(raw: str) -> tuple[bool, str | None]:
    """Strict contract: valid iff both tag pairs are present. Code = <answer> body."""
    think, answer = THINK_RE.search(raw), ANSWER_RE.search(raw)
    if not answer:
        return False, None
    code = answer.group(1).strip("\n")
    fenced = FENCE_RE.match(code)  # tolerate a stray fence; it is not counted as an error
    if fenced:
        code = fenced.group(1)
    return bool(think), code


def extract_fallback_code(raw: str) -> str | None:
    """Secondary metric only: last fenced code block when there is no <answer> block."""
    blocks = LAST_PY_FENCE_RE.findall(raw)
    return blocks[-1] if blocks else None


# Untrusted code runs as `nobody` with an 8 GiB address-space cap. `prlimit` and the
# user/group arguments avoid preexec_fn, which is unsafe in a thread pool.
_SANDBOX_USER = {"user": 65534, "group": 65534, "extra_groups": []} if os.geteuid() == 0 else {}


def run_python(source: str, stdin: str = "", timeout: float = 10.0) -> tuple[str, str, str]:
    """Run source in a fresh interpreter. Returns (status, stdout, stderr)."""
    with tempfile.TemporaryDirectory() as tmp:
        os.chmod(tmp, 0o777)
        path = Path(tmp) / "prog.py"
        path.write_text(source)
        os.chmod(path, 0o644)
        try:
            p = subprocess.run(
                ["prlimit", f"--as={8 << 30}", sys.executable, str(path)],
                input=stdin,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=tmp,
                start_new_session=True,
                env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONNOUSERSITE": "1", "HOME": tmp},
                **_SANDBOX_USER,
            )
        except subprocess.TimeoutExpired:
            return "timeout", "", ""
        return ("ok" if p.returncode == 0 else "error"), p.stdout, p.stderr[-2000:]


def _last_error_line(stderr: str) -> str:
    lines = [l for l in stderr.strip().splitlines() if l.strip()]
    return lines[-1] if lines else ""


def _outputs_match(got: str, expected: str) -> bool:
    """LiveCodeBench stdout rule: compare stripped lines, then tokens as decimals."""
    g = [l.strip() for l in got.strip().splitlines()]
    e = [l.strip() for l in expected.strip().splitlines()]
    if g == e:
        return True
    if len(g) != len(e):
        return False
    from decimal import Decimal, InvalidOperation

    for gl, el in zip(g, e):
        gt, et = gl.split(), el.split()
        if len(gt) != len(et):
            return False
        try:
            if [Decimal(x) for x in gt] != [Decimal(x) for x in et]:
                return False
        except InvalidOperation:
            return False
    return True


# Same import prelude the official LiveCodeBench harness prepends to every solution.
LCB_PRELUDE = """\
from string import *
from re import *
from datetime import *
from collections import *
from heapq import *
from bisect import *
from copy import *
from math import *
from random import *
from statistics import *
from itertools import *
from functools import *
from operator import *
from io import *
from sys import *
from json import *
from builtins import *
from typing import *
import string, re, datetime, collections, heapq, bisect, copy, math, random, statistics, itertools, functools, operator, io, sys, json
sys.setrecursionlimit(50000)
"""

FUNC_HARNESS = """
{code}

import json as _json
_args = [_json.loads(_l) for _l in {inp!r}.split("\\n")]
_out = getattr(Solution(), {func!r})(*_args)
print(_json.dumps(_out))
"""


def run_tests(code: str, task: dict, timeout: float) -> dict:
    """Syntax-check code, then run the task's tests. Returns the test-related result fields."""
    result = {"syntax_ok": True, "passed": False, "passed_test_cases": 0, "total_test_cases": None,
              "error_type": None, "error_message": None}
    try:
        compile(code, "<answer>", "exec")
    except SyntaxError as e:
        return {**result, "syntax_ok": False, "error_type": "syntax_error", "error_message": str(e)}

    if "tests" not in task:  # MBPP / MBPP+: one script with all asserts
        result["total_test_cases"] = 1
        status, _, err = run_python(code + "\n\n" + task["test_code"], timeout=timeout)
        if status == "ok":
            return {**result, "passed": True, "passed_test_cases": 1}
        etype = "timeout" if status == "timeout" else ("wrong_answer" if "AssertionError" in err else "runtime_error")
        return {**result, "error_type": etype, "error_message": _last_error_line(err) or None}

    tests = task["tests"]  # LiveCodeBench: one process per test case, stop at first failure
    result["total_test_cases"] = len(tests)
    for t in tests:
        if task["execution_mode"] == "stdin":
            status, out, err = run_python(LCB_PRELUDE + code, stdin=t["input"], timeout=timeout)
            ok = status == "ok" and _outputs_match(out, t["output"])
        else:
            src = LCB_PRELUDE + FUNC_HARNESS.format(code=code, inp=t["input"], func=task["func_name"])
            status, out, err = run_python(src, timeout=timeout)
            try:
                ok = status == "ok" and json.loads(out.strip().splitlines()[-1]) == json.loads(t["output"])
            except (json.JSONDecodeError, IndexError):
                ok = False
        if not ok:
            etype = {"timeout": "timeout", "error": "runtime_error"}.get(status, "wrong_answer")
            return {**result, "error_type": etype, "error_message": _last_error_line(err) or None}
        result["passed_test_cases"] += 1
    return {**result, "passed": True}


def grade_sample(raw: str, task: dict, timeout: float) -> dict:
    """Grade one completion.

    `passed` (primary) grades only code inside <answer>...</answer>; `format_valid` separately
    records whether both tag pairs are present. `fallback_passed` (secondary) also accepts
    the last fenced code block when there is no <answer> block.
    """
    fmt_ok, code = extract_code(raw)
    if code is not None:
        tested = run_tests(code, task, timeout)
        return {"format_valid": fmt_ok, "extracted_code": code, **tested, "fallback_passed": tested["passed"]}
    fallback = extract_fallback_code(raw)
    return {"format_valid": False, "extracted_code": None, "syntax_ok": None, "passed": False,
            "passed_test_cases": 0, "total_test_cases": None, "error_type": "format_error",
            "error_message": "no <answer>...</answer> block",
            "fallback_passed": run_tests(fallback, task, timeout)["passed"] if fallback else False}


def grade_all(raw_records: list[dict], tasks: list[dict], timeout: float, workers: int = 64) -> list[dict]:
    by_id = {t["task_id"]: t for t in tasks}

    def one(r):
        g = grade_sample(r["raw_output"], by_id[r["task_id"]], timeout)
        return {"task_id": r["task_id"], "difficulty": r["difficulty"], "sample_idx": r["sample_idx"], **g}

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, raw_records))


# ---------------------------------------------------------------------------
# Metrics and storage
# ---------------------------------------------------------------------------


def summarize(run_id: str, benchmark: str, k: int, raw: list[dict], evals: list[dict], wall: float) -> dict:
    raw_df, ev = pd.DataFrame(raw), pd.DataFrame(evals)
    per_task = ev.groupby("task_id")["passed"].mean()  # c_i / k
    extracted = ev[ev["extracted_code"].notna()]
    total_tokens = int(raw_df["total_tokens"].sum())
    n_correct = int(ev["passed"].sum())
    metrics = {
        f"avg_at_{k}": round(per_task.mean(), 4),
        "pass_at_1": round(per_task.mean(), 4),  # unbiased pass@1 from k samples = mean(c_i / k)
        f"avg_at_{k}_fallback": round(ev.groupby("task_id")["fallback_passed"].mean().mean(), 4),  # secondary
        "format_compliance_rate": round(ev["format_valid"].mean(), 4),
        "syntax_error_rate": round((extracted["syntax_ok"] == False).mean(), 4) if len(extracted) else None,  # noqa: E712
        "truncation_rate": round((raw_df["finish_reason"] == "length").mean(), 4),
        "mean_reasoning_tokens": round(raw_df["reasoning_tokens"].mean(), 1),
        "mean_answer_tokens": round(raw_df["answer_tokens"].mean(), 1),
        "mean_total_tokens": round(raw_df["total_tokens"].mean(), 1),
        "tokens_per_correct_solution": round(total_tokens / n_correct, 1) if n_correct else None,
        "mean_latency_sec": round(raw_df["latency_sec"].mean(), 3) if raw_df["latency_sec"].notna().any() else None,
        "total_wall_clock_sec": round(wall, 1),
        "tokens_per_sec": round(total_tokens / wall, 1),
    }
    if ev["difficulty"].notna().any():
        diff = ev.groupby(["difficulty", "task_id"])["passed"].mean().groupby("difficulty").mean()
        metrics[f"avg_at_{k}_by_difficulty"] = {d: round(v, 4) for d, v in diff.items()}
    return {"run_id": run_id, "benchmark": benchmark, "num_tasks": int(per_task.size), "samples_per_task": k,
            "metrics": metrics}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    tmp.replace(path)  # atomic: a half-written file never looks complete


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f]


def write_checksums(run_dir: Path) -> None:
    lines = []
    for p in sorted(run_dir.rglob("*")):
        if p.is_file() and p.name != "SHA256SUMS":
            lines.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(run_dir)}")
    (run_dir / "SHA256SUMS").write_text("\n".join(lines) + "\n")


def verify_run(run_dir: Path, plan: dict[str, int], sample_ids: dict[str, list[str]]) -> pd.DataFrame:
    """Acceptance checks. Raises AssertionError on the first failure."""
    rows = []
    for bench, k in plan.items():
        ids = sample_ids[bench]
        assert len(ids) == len(set(ids)) == 100, f"{bench}: expected 100 unique task ids"
        for name in ["raw_generations.jsonl", "eval_results.jsonl"]:
            recs = read_jsonl(run_dir / bench / name)
            keys = Counter((r["task_id"], r["sample_idx"]) for r in recs)
            expected = {(t, i) for t in ids for i in range(k)}
            assert set(keys) == expected, f"{bench}/{name}: task/sample set differs from plan"
            assert max(keys.values()) == 1, f"{bench}/{name}: duplicate records"
            assert len(recs) == 100 * k
        summary = json.loads((run_dir / bench / "summary.json").read_text())
        assert summary["num_tasks"] == 100 and summary["samples_per_task"] == k
        rows.append({"benchmark": bench, "tasks": 100, "k": k, "records": 100 * k, "status": "OK"})
    return pd.DataFrame(rows)
