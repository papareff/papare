# Multi-Agent Debate and Retrieval Classification Code

This repository contains the code portion of a research project on text classification with:

- sample-aware routing between zero-shot and few-shot inference
- TF-IDF retrieval for in-context examples
- multi-agent debate with differentiated review lenses
- judge-based voting, confidence gating, and ablation scripts

Large artifacts are intentionally excluded from this code embed: trained models, checkpoints, datasets, cached environments, PPT files, and experiment result tables.

## Structure

- `src/agents/` - base generation wrapper, debater agent, and judge agent
- `src/pipeline/` - end-to-end inference pipeline
- `src/retriever/` - few-shot example retriever
- `src/router/` - sample-aware strategy router
- `src/utils/` - dataset loading helpers
- `scripts/` - data preparation, training, evaluation, diagnostics, and ablation scripts
- `configs/` - base configuration

## Setup

Install non-PyTorch dependencies with:

```bash
pip install -r requirements.txt
```

The original project expects a CUDA-enabled PyTorch installation. Install PyTorch separately according to the target machine and CUDA version.

## Notes

Most scripts assume they are run from the repository root and write outputs under `experiments/results/`. Model directories such as `models/irony_distilbert` or `models/judge_roberta_cw_samedomain_ablation` must be trained or restored separately before running full evaluations.
