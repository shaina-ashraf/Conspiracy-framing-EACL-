# Conspiracy Frame

This repository contains the data, code, and analysis artifacts accompanying the EACL 2027 Findings paper on conspiracy-theory classification and conspiracy-frame span extraction in Telegram messages.

The public release excludes Telegram channel identifiers, raw crawls, generated model outputs, logs, and model caches. The released annotation files use pseudonymous annotator identifiers (`A001`--`A013`).

> **Content warning:** The dataset contains conspiratorial, hateful, discriminatory, and otherwise offensive language originating from public Telegram messages.

## Repository Structure

- `llm_annotate.py`: runs LLM annotation on the annotated in-domain dataset.
- `llm_annotate_unseen.py`: runs LLM annotation on the out-of-domain/unseen dataset.
- `jobs/`: SLURM scripts used to run model annotation experiments.
- `Notebooks/`: notebooks for annotation analysis, model comparison, span analysis, and unseen-data analysis.
- `publication/`: paper-facing prompts, tables, figures, and analysis notebooks.
- `scripts/`: utility scripts for organizing experiment outputs.

## Released Data

- `Data/all_classifications.csv`: 3,243 annotations over 2,077 texts for conspiracy classification and supporting labels.
- `Data/all_span_annotations.csv`: 934 annotations over 755 texts for the five Conspiracy Frame elements (`plan`, `secret`, `out-group`, `in-group`, and `call-to-action`).
- `Data/all_span_annotations_with_frames.csv`: the span annotations augmented with the available FrameNet mappings for `plan` and `secret`.

The FrameNet-enriched file does not contain mappings for the other three Conspiracy Frame elements because those mappings are not present in the current source data.

## Experiments

The code supports two main tasks:

1. Binary classification: predict whether a message is conspiratorial.
2. Span extraction: identify conspiracy-frame elements such as plan/event, secret, out-group, in-group, and call-to-action.

The experiments compare zero-shot, few-shot, and frame-guided prompting settings for Llama models on in-domain annotated data and out-of-domain unseen data.

## Additional Local Inputs

The released annotation files are included under `Data/`. Some experiment scripts additionally expect prompts and out-of-domain inputs under paths such as:

- `Data/prompts/`
- `Data/unseen/`

Generated model outputs are written under `outputs/`. These additional paths remain excluded from version control.

## Reproduction Notes

The SLURM scripts in `jobs/` show the model, prompt, and output-file settings used for the experiments. The notebooks in `publication/` and `Notebooks/` aggregate the resulting predictions into the reported tables and figures.
