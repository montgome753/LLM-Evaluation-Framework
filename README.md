<div align="center">

# 🧠 LLM Evaluation Framework

### Production-Grade Benchmarking for Any LLM

[![CI](https://github.com/vignesh2027/LLM-Evaluation-Framework/actions/workflows/ci.yml/badge.svg)](https://github.com/vignesh2027/LLM-Evaluation-Framework/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI version](https://img.shields.io/badge/pypi-1.0.0-orange.svg)](https://pypi.org/project/llm-evaluation-framework/)
[![HuggingFace](https://img.shields.io/badge/🤗-Dataset-yellow)](https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-FF4B4B)](https://vignesh2027.github.io/LLM-Evaluation-Framework/)
[![Stars](https://img.shields.io/github/stars/vignesh2027/LLM-Evaluation-Framework?style=social)](https://github.com/vignesh2027/LLM-Evaluation-Framework/stargazers)

**Evaluate GPT-4, Claude, Gemini, Mistral & Llama on accuracy, latency, cost,
hallucination rate, and reasoning quality — with a beautiful Streamlit dashboard,
REST API, CLI, and PDF reports.**

[🚀 Quick Start](#-quick-start) · [📊 Dashboard](#-dashboard) · [📖 Docs](https://vignesh2027.github.io/LLM-Evaluation-Framework/) · [🤗 Dataset](https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark) · [⭐ Star Us](#)

![Demo GIF Placeholder](https://placehold.co/900x450/1a1a2e/e94560?text=LLM+Evaluation+Framework+Demo)

</div>

---

## ✨ Why This Framework?

> *"You can't improve what you can't measure."*

Most LLM benchmarking tools evaluate a single model in isolation. This framework evaluates **any model against any model**, with full async support, cost tracking, and hallucination detection — all in one place.

| Feature | Description |
|--------|-------------|
| 🎯 **5 Metrics** | Accuracy, Latency (p50/p95/p99), Cost, Hallucination Rate, Reasoning Score |
| 🏆 **Built-in Benchmarks** | MMLU (57 subjects), TruthfulQA (817 questions), Custom CSV/JSON |
| ⚡ **Full Async** | Parallel evaluation with configurable concurrency |
| 📊 **Beautiful Dashboard** | Radar charts, latency histograms, cost-vs-quality scatter plots |
| 🔌 **10+ Models** | GPT-4o, Claude 3.5, Gemini 1.5, Mistral, Llama 3 via LiteLLM |
| 🌐 **REST API** | FastAPI with OpenAPI docs at `/docs` |
| 💻 **CLI** | `llm-eval run --model gpt-4o --benchmark mmlu --samples 100` |
| 📄 **PDF Reports** | Auto-generated professional evaluation reports |
| 🐳 **Docker** | Single-command deployment |
| ✅ **CI/CD** | GitHub Actions with test coverage |

---

## 🏗 Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM Evaluation Framework                        │
│                                                                     │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────────┐    │
│  │   CLI    │   │  FastAPI │   │Streamlit │   │ PDF Reports  │    │
│  │llm-eval  │   │   /docs  │   │Dashboard │   │  Generator   │    │
│  └────┬─────┘   └────┬─────┘   └────┬─────┘   └──────┬───────┘    │
│       └──────────────┴──────────────┴────────────────┘             │
│                               │                                     │
│                    ┌──────────▼──────────┐                          │
│                    │   Core Evaluator    │                          │
│                    │  (Async Engine)     │                          │
│                    └──────────┬──────────┘                          │
│                               │                                     │
│          ┌────────────────────┼────────────────────┐               │
│          │                    │                    │               │
│  ┌───────▼──────┐   ┌────────▼──────┐   ┌────────▼──────┐        │
│  │   Metrics    │   │  Benchmarks   │   │   Database    │        │
│  │              │   │               │   │   (SQLite)    │        │
│  │ • Accuracy   │   │ • MMLU        │   │               │        │
│  │ • Latency    │   │ • TruthfulQA  │   │ • Results     │        │
│  │ • Cost       │   │ • Custom CSV  │   │ • Export CSV  │        │
│  │ • Hallucin.  │   │               │   │ • Export JSON │        │
│  │ • Reasoning  │   └───────────────┘   └───────────────┘        │
│  └──────────────┘                                                   │
│                               │                                     │
│                    ┌──────────▼──────────┐                          │
│                    │     LiteLLM         │                          │
│                    │  (Unified API)      │                          │
│                    └──────────┬──────────┘                          │
│                               │                                     │
│   ┌────────┐  ┌────────┐  ┌──────────┐  ┌─────────┐  ┌────────┐  │
│   │ OpenAI │  │Anthropic│  │  Google  │  │ Mistral │  │  Meta  │  │
│   │ GPT-4o │  │ Claude  │  │  Gemini  │  │         │  │ Llama3 │  │
│   └────────┘  └────────┘  └──────────┘  └─────────┘  └────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install

```bash
# From PyPI
pip install llm-evaluation-framework

# From source
git clone https://github.com/vignesh2027/LLM-Evaluation-Framework.git
cd LLM-Evaluation-Framework
pip install -e .
```

### 2. Set API Keys

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY, ANTHROPIC_API_KEY, etc.
```

### 3. Run Your First Evaluation

```bash
# Evaluate GPT-4o-mini on MMLU (100 samples)
llm-eval run --model gpt-4o-mini --benchmark mmlu --samples 100

# Compare 3 models side-by-side
llm-eval compare \
  --models gpt-4o-mini \
  --models claude-3-haiku-20240307 \
  --models gemini/gemini-1.5-flash \
  --benchmark mmlu --samples 50

# Launch the Streamlit dashboard
llm-eval dashboard

# Start the REST API
llm-eval serve --port 8000
```

---

## 📊 Dashboard

```bash
streamlit run llm_eval/dashboard/app.py
# → http://localhost:8501
```

The dashboard includes:
- **Radar Chart** — multi-dimensional model comparison
- **Latency Histogram** — response time distribution
- **Cost vs Quality Scatter Plot** — value analysis
- **Side-by-side comparison** — run same samples on N models
- **CSV/JSON upload** — custom benchmark support
- **PDF report download** — one-click export

---

## 🌐 REST API

```bash
uvicorn llm_eval.api.main:app --reload
# → http://localhost:8000/docs
```

### Key Endpoints

```http
POST /evaluate          Evaluate a model on a benchmark
POST /compare           Compare multiple models side-by-side
POST /evaluate/custom   Upload CSV/JSON for custom evaluation
GET  /results           List all stored results
GET  /results/{run_id}  Get a specific run
GET  /export/csv        Download results as CSV
GET  /export/json       Download results as JSON
POST /report            Generate PDF report
GET  /models            List supported models & pricing
GET  /benchmarks        List available benchmarks
GET  /health            Health check
```

### Example

```python
import httpx

# Run evaluation
resp = httpx.post("http://localhost:8000/evaluate", json={
    "model": "gpt-4o-mini",
    "benchmark": "mmlu",
    "num_samples": 50,
    "temperature": 0.0
})
result = resp.json()
print(f"Accuracy: {result['accuracy']:.1%}")
print(f"Avg Latency: {result['avg_latency_ms']:.0f}ms")
print(f"Cost: ${result['total_cost_usd']:.4f}")
```

---

## 💻 CLI Reference

```
Usage: llm-eval [OPTIONS] COMMAND [ARGS]...

  LLM Evaluation & Benchmarking Framework

Commands:
  run        Run a single model evaluation
  compare    Compare multiple models side-by-side
  results    List stored evaluation results
  export     Export results to CSV or JSON
  report     Generate a PDF evaluation report
  serve      Start the FastAPI server
  dashboard  Launch the Streamlit dashboard

Options:
  -v, --verbose  Enable debug logging
  --version      Show version and exit
```

---

## 🐍 Python API

```python
import asyncio
from llm_eval.core.evaluator import LLMEvaluator, EvaluationConfig
from llm_eval.benchmarks.mmlu import MMLUBenchmark

async def main():
    evaluator = LLMEvaluator()
    
    # Load benchmark samples
    samples = MMLUBenchmark().load(num_samples=100)
    
    # Configure evaluation
    config = EvaluationConfig(
        model="gpt-4o-mini",
        benchmark="mmlu",
        num_samples=100,
        temperature=0.0,
        concurrency=10,
    )
    
    # Run evaluation
    result = await evaluator.evaluate(config, samples)
    
    print(f"Model: {result.model}")
    print(f"Accuracy: {result.accuracy:.2%}")
    print(f"Avg Latency: {result.avg_latency_ms:.0f}ms")
    print(f"P95 Latency: {result.p95_latency_ms:.0f}ms")
    print(f"Total Cost: ${result.total_cost_usd:.4f}")
    print(f"Hallucination Rate: {result.hallucination_rate:.2%}")
    print(f"Reasoning Score: {result.avg_reasoning_score:.1f}/10")

asyncio.run(main())
```

### Side-by-Side Comparison

```python
# Compare 3 models on the same samples
configs = [
    EvaluationConfig(model="gpt-4o-mini", benchmark="mmlu", num_samples=50),
    EvaluationConfig(model="claude-3-haiku-20240307", benchmark="mmlu", num_samples=50),
    EvaluationConfig(model="gemini/gemini-1.5-flash", benchmark="mmlu", num_samples=50),
]

results = await evaluator.evaluate_multiple(configs, samples)
for r in sorted(results, key=lambda x: x.accuracy, reverse=True):
    print(f"{r.model}: {r.accuracy:.1%} accuracy, {r.avg_latency_ms:.0f}ms, ${r.total_cost_usd:.4f}")
```

---

## 🐳 Docker

```bash
# Start both API + Dashboard
docker-compose up -d

# API at http://localhost:8000/docs
# Dashboard at http://localhost:8501
```

---

## 📦 Supported Models

| Provider | Models |
|----------|--------|
| **OpenAI** | gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo, o1, o1-mini |
| **Anthropic** | claude-3-5-sonnet, claude-3-5-haiku, claude-3-opus, claude-3-haiku |
| **Google** | gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash |
| **Mistral** | mistral-large, mistral-small, open-mistral-7b |
| **Meta (Llama)** | Llama-3-70b, Llama-3-8b (via Together AI) |
| **Any** | Any [LiteLLM-compatible](https://docs.litellm.ai/docs/providers) model |

---

## 📊 Benchmarks

| Benchmark | Samples | Description |
|-----------|---------|-------------|
| **MMLU** | ~14,000 | Massive Multitask Language Understanding — 57 academic subjects |
| **TruthfulQA** | 817 | Tests factual truthfulness, designed to catch common misconceptions |
| **Custom** | Any | Upload your own CSV or JSON with `prompt` + `expected` columns |

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=llm_eval --cov-report=html

# Run specific test class
pytest tests/test_evaluator.py::TestAccuracyMetric -v
```

---

## 🤗 HuggingFace Dataset

The evaluation benchmark dataset is available on HuggingFace:

```python
from datasets import load_dataset
ds = load_dataset("vigneshwar234/llm-eval-benchmark")
```

Dataset includes MMLU + TruthfulQA samples with evaluation metadata.

---

## 📁 Project Structure

```
LLM-Evaluation-Framework/
├── llm_eval/
│   ├── core/
│   │   └── evaluator.py       ← Main async evaluation engine
│   ├── metrics/
│   │   ├── accuracy.py        ← Multi-strategy accuracy scorer
│   │   ├── hallucination.py   ← Hallucination + reasoning quality
│   │   ├── latency.py         ← Percentile stats & SLA tracking
│   │   └── cost.py            ← Token cost calculator (10+ providers)
│   ├── benchmarks/
│   │   ├── mmlu.py            ← MMLU loader (HF + local cache + builtin)
│   │   ├── truthfulqa.py      ← TruthfulQA loader
│   │   └── custom.py          ← CSV/JSON custom benchmark loader
│   ├── dashboard/
│   │   └── app.py             ← Streamlit dashboard (5 pages)
│   ├── api/
│   │   └── main.py            ← FastAPI REST API (10 endpoints)
│   ├── cli/
│   │   └── main.py            ← Click CLI (7 commands)
│   ├── reports/
│   │   └── generator.py       ← ReportLab PDF generator
│   └── database/
│       └── models.py          ← SQLite persistence layer
├── tests/
│   └── test_evaluator.py      ← 40+ pytest tests
├── .github/workflows/
│   └── ci.yml                 ← CI: test + lint + Docker + Pages deploy
├── docs/
│   └── index.html             ← GitHub Pages site
├── requirements.txt
├── .env.example
├── Dockerfile
├── docker-compose.yml
└── setup.py
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run `pytest` and `ruff check`
5. Submit a Pull Request

---

## 📜 License

MIT License — see [LICENSE](LICENSE) for details.

---

## ⭐ Star History

If this project helps you, please **star it** — it helps others discover it!

[![Star History Chart](https://api.star-history.com/svg?repos=vignesh2027/LLM-Evaluation-Framework&type=Date)](https://star-history.com/#vignesh2027/LLM-Evaluation-Framework&Date)

---

<div align="center">
Made with ❤️ by <a href="https://github.com/vignesh2027">vignesh2027</a>
<br>
<a href="https://github.com/vignesh2027/LLM-Evaluation-Framework">GitHub</a> ·
<a href="https://vignesh2027.github.io/LLM-Evaluation-Framework/">Docs</a> ·
<a href="https://huggingface.co/datasets/vigneshwar234/llm-eval-benchmark">HuggingFace</a>
</div>
