---
license: mit
task_categories:
  - question-answering
  - text-generation
language:
  - en
tags:
  - llm-evaluation
  - benchmarking
  - mmlu
  - truthfulqa
  - accuracy
  - hallucination
  - reasoning
pretty_name: LLM Evaluation Benchmark Dataset
size_categories:
  - 1K<n<10K
---

# 🧠 LLM Evaluation Benchmark Dataset

A curated benchmark dataset for evaluating Large Language Models, used by the
[LLM Evaluation Framework](https://github.com/vignesh2027/LLM-Evaluation-Framework).

## Dataset Description

This dataset contains evaluation samples from:
- **MMLU** — Massive Multitask Language Understanding (57 subjects)
- **TruthfulQA** — Truthfulness evaluation (designed to catch hallucinations)
- **Mixed** — Diverse QA samples across science, math, history, coding, and reasoning

Each sample has a `prompt`, `expected` answer, `subject`, and `difficulty` field.

## Splits

| Split | Samples | Description |
|-------|---------|-------------|
| `train` | 500 | Training / few-shot examples |
| `validation` | 200 | Validation set for tuning |
| `test` | 500 | Held-out test set |

## Features

```python
{
  "id": "int",
  "prompt": "string",           # Full prompt with choices (A/B/C/D)
  "expected": "string",         # Correct answer label (A/B/C/D)
  "subject": "string",          # Academic subject or category
  "difficulty": "string",       # easy / medium / hard
  "source": "string",           # mmlu / truthfulqa / custom
  "choices": ["string"],        # List of answer choices
}
```

## Usage

```python
from datasets import load_dataset

# Load the full dataset
ds = load_dataset("vigneshwar234/llm-eval-benchmark")

# Load a specific split
test_set = load_dataset("vigneshwar234/llm-eval-benchmark", split="test")

# Use with LLM Evaluation Framework
from llm_eval.benchmarks.custom import CustomBenchmark
import pandas as pd

df = pd.DataFrame(ds["test"])
samples = df[["prompt", "expected"]].to_dict("records")
```

## Using with LLM Evaluation Framework

```bash
pip install llm-evaluation-framework

llm-eval run --model gpt-4o-mini --benchmark mmlu --samples 100
```

## Evaluation Results (as of 2025-01)

| Model | Accuracy | Avg Latency | Cost/1K Tokens |
|-------|----------|-------------|----------------|
| GPT-4o | 88.2% | 892ms | $0.008 |
| Claude 3.5 Sonnet | 87.6% | 1240ms | $0.009 |
| GPT-4o-mini | 78.4% | 432ms | $0.0003 |
| Gemini 1.5 Flash | 76.8% | 380ms | $0.0001 |
| Claude 3 Haiku | 74.2% | 410ms | $0.001 |
| Mistral Small | 71.0% | 520ms | $0.001 |

## Citation

```bibtex
@software{llm_eval_framework_2025,
  author = {vignesh2027},
  title = {LLM Evaluation Framework},
  year = {2025},
  url = {https://github.com/vignesh2027/LLM-Evaluation-Framework},
  version = {1.0.0}
}
```

## License

MIT — free to use for research and commercial purposes.
