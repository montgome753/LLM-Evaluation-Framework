"""Setup script for LLM Evaluation Framework."""
from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

with open("requirements.txt") as f:
    requirements = [
        line.strip()
        for line in f
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="llm-evaluation-framework",
    version="1.0.0",
    author="vignesh2027",
    author_email="lkvarnesh@gmail.com",
    description="Production-grade LLM Evaluation & Benchmarking Framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vignesh2027/LLM-Evaluation-Framework",
    project_urls={
        "Bug Tracker": "https://github.com/vignesh2027/LLM-Evaluation-Framework/issues",
        "Documentation": "https://vignesh2027.github.io/LLM-Evaluation-Framework/",
        "HuggingFace": "https://huggingface.co/datasets/vignesh2027/llm-eval-benchmark",
    },
    packages=find_packages(exclude=["tests*"]),
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dashboard": ["streamlit>=1.40.0", "plotly>=5.24.0", "pandas>=2.2.0"],
        "reports": ["reportlab>=4.2.0"],
        "dev": ["pytest", "pytest-asyncio", "pytest-cov", "mypy", "ruff"],
    },
    entry_points={
        "console_scripts": [
            "llm-eval=llm_eval.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Testing",
    ],
    keywords=[
        "llm", "evaluation", "benchmark", "gpt", "claude", "gemini",
        "mistral", "mmlu", "truthfulqa", "ai", "nlp", "litellm",
    ],
)
