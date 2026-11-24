# Conspiracy-Framing

This repository contains code and analysis artifacts for experiments on conspiracy-theory classification and conspiracy-frame span extraction in Telegram messages.

The repository is organized for anonymous review. Raw data files, generated model outputs, logs, and model caches are not included in GitHub. The scripts expect those files in local `Data/` and `outputs/` directories when reproducing the experiments.

## Repository Structure

- `llm_annotate.py`: runs LLM annotation on the annotated in-domain dataset.
- `llm_annotate_unseen.py`: runs LLM annotation on the out-of-domain/unseen dataset.
- `jobs/`: SLURM scripts used to run model annotation experiments.
- `Notebooks/`: notebooks for annotation analysis, model comparison, span analysis, and unseen-data analysis.
- `publication/`: paper-facing prompts, tables, figures, and analysis notebooks.
- `scripts/`: utility scripts for organizing experiment outputs.

## Experiments

The code supports two main tasks:

1. Binary classification: predict whether a message is conspiratorial.
2. Span extraction: identify conspiracy-frame elements such as plan/event, secret, out-group, in-group, and call-to-action.

The experiments compare zero-shot, few-shot, and frame-guided prompting settings for Llama models on in-domain annotated data and out-of-domain unseen data.

## Expected Local Inputs

The main scripts expect project data and prompts under `Data/`, for example:

- `Data/all_classifications.csv`
- `Data/prompts/`
- `Data/unseen/`

Generated model outputs are written under `outputs/`. These paths are excluded from version control because they can contain large generated files or data that should not be redistributed through the code repository.

## Reproduction Notes

The SLURM scripts in `jobs/` show the model, prompt, and output-file settings used for the experiments. The notebooks in `publication/` and `Notebooks/` aggregate the resulting predictions into the reported tables and figures.
