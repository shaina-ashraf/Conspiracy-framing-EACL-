#!/usr/bin/env python3
"""
Annotate samples with LLM models (default: Llama-3-8B-Instruct) using the few-shot prompt.

- Reads few-shot prompt from Data/prompts/llama_few_shot_prompt_unseen.txt
- Reads texts from CSV (default Data/unseen/un-labelled.csv)
- For each text: expects JSON with:
    is_conspiratorial (bool)
    rationale_short (str)
    confidence (float)
    spans (list of {label,text,start,end}) — only when conspiratorial
- Optionally writes JSONL if --out is provided (disabled by default)
- Writes a copy of the CSV with LLM columns merged in (by row index)
- Writes human-schema-mirrored CSVs (supportive column left blank intentionally):
    Data/llm_all_classifications.csv (annotator_id,text_id,annotation_id,text,conspiracy,supportive,misinformative,confidence,rationale_short)
    Data/llm_all_span_annotations.csv (annotator_id,text_id,annotation_id,text,plan_event,secret,out_group,in_group,call_to_action)
"""
import os
import re
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Any

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers import BitsAndBytesConfig

    BITSANDBYTES_AVAILABLE = True
except ImportError:
    BITSANDBYTES_AVAILABLE = False


DEFAULT_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
DEFAULT_PROMPT_PATH = "Data/prompts/llama_few_shot_prompt_unseen.txt"
DEFAULT_CSV_PATH = "Data/unseen/un-labelled.csv"
DEFAULT_OUTPUT_PATH = None  # JSONL output disabled by default
DEFAULT_UPDATED_CSV = "outputs/llama8b-fewshot_unseen_annotations_raw.csv"
DEFAULT_FAIL_LOG = "outputs/llama8b-fewshot_annotation_failures.jsonl"
DEFAULT_LIMIT = None  # None means process all unique texts
VALID_LABELS = {"plan_event", "secret", "out_group", "in_group", "call_to_action"}
PUNCT = "\"'""'`,.;:!?()[]{}"

# Output paths - will be adjusted based on prompt filename
LLM_ALL_CLASS_CSV = "outputs/llama8b-fewshot_all_classifications.csv"
LLM_ALL_SPANS_CSV = "outputs/llama8b-fewshot_all_span_annotations.csv"


def get_output_prefix_from_prompt(prompt_path: str, model_name: str | None = None) -> str:
    """Detect prompt type from filename and model name, return appropriate output prefix."""
    prompt_name = os.path.basename(prompt_path).lower()

    # Detect model size from model name/path
    model_prefix = "llama8b"  # default
    if model_name:
        model_name_lower = model_name.lower()
        if ("3.3" in model_name_lower and ("70" in model_name_lower or "70b" in model_name_lower)) or (
            "70b" in model_name_lower or "70-b" in model_name_lower
        ):
            model_prefix = "llama-3.3-70B"
        elif "70b" in model_name_lower and "3.3" not in model_name_lower:
            model_prefix = "llama70b"
        elif "8b" in model_name_lower or "8-b" in model_name_lower:
            model_prefix = "llama8b"

    prompt_type = "fewshot" if "fewshot" in prompt_name or "few_shot" in prompt_name else "prompt"
    prompt_suffix = "-frames" if "frame" in prompt_name else ""
    return f"{model_prefix}-{prompt_type}{prompt_suffix}"


def slugify_filename(name: str) -> str:
    slug = re.sub(r"[^\w\-]+", "-", name.lower()).strip("-")
    return slug or "data"


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def detect_text_col(df: pd.DataFrame) -> str:
    for cand in ["text", "content", "body", "post", "article_text"]:
        if cand in df.columns:
            return cand
    raise ValueError("No suitable text column found in CSV. Expected one of: text, content, body, post, article_text")


def load_texts(csv_path: str, limit: int | None = DEFAULT_LIMIT) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    """
    Load texts, but only unique text_ids (LLM is a single annotator, should annotate each text once).
    If limit is None, process all unique texts.
    """
    df = pd.read_csv(csv_path)
    text_col = detect_text_col(df)
    df = df.reset_index(drop=False).rename(columns={"index": "row_index"})
    df = df[df[text_col].astype(str).str.len() > 0]

    if "text_id" in df.columns:
        df_unique = df.drop_duplicates(subset="text_id", keep="first")
        print(f"Filtered to unique text_ids: {len(df)} -> {len(df_unique)} rows")
        df = df_unique

    if limit is None:
        subset = df[["row_index", text_col]].rename(columns={text_col: "text"})
        text_ids = df["text_id"].values if "text_id" in df.columns else None
        print(f"Processing all {len(df)} unique texts (limit=None)")
    else:
        subset = df.head(limit)[["row_index", text_col]].rename(columns={text_col: "text"})
        text_ids = df.head(limit)["text_id"].values if "text_id" in df.columns else None
        print(f"Processing {len(subset)} unique texts (limit={limit})")

    samples = subset.to_dict(orient="records")
    if "text_id" in df.columns and text_ids is not None:
        for i, sample in enumerate(samples):
            sample["text_id"] = text_ids[i] if i < len(text_ids) else None

    return df, samples, text_col


def build_chat_prompt(system_prompt: str, text: str) -> List[Dict[str, str]]:
    if "{{INPUT_TEXT}}" in system_prompt or "{INPUT_TEXT}" in system_prompt:
        user_block = system_prompt.replace("{{INPUT_TEXT}}", text).replace("{INPUT_TEXT}", text)
        return [{"role": "user", "content": user_block}]
    else:
        user_block = f"Input text\n{text}\n\nReturn only the JSON."
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_block},
        ]


def _last_top_level_json(s: str) -> Dict[str, Any]:
    starts: List[int] = []
    pairs: List[Tuple[int, int]] = []
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                starts.append(i)
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and starts:
                    pairs.append((starts[-1], i))
                    starts.pop()
    for first, last in reversed(pairs):
        try:
            return json.loads(s[first : last + 1])
        except Exception:
            continue
    matches = list(re.finditer(r"\{[\s\S]*?\}", s))
    if matches:
        for mm in reversed(matches):
            try:
                return json.loads(mm.group(0))
            except Exception:
                continue
    return {}


def _trim_quotes_punct(s: str) -> str:
    return s.strip().strip(PUNCT).strip()


def _try_snap_span(src: str, text: str, start: int, end: int, window: int = 12) -> Tuple[int, int, bool]:
    n = len(src)
    t = _trim_quotes_punct(text)
    if not t:
        return start, end, False
    if 0 <= start <= end <= n and src[start:end] == t:
        return start, end, True
    for delta in range(0, 4):
        for sgn in (-1, 1):
            s2 = max(0, start + sgn * delta)
            e2 = min(n, end + sgn * delta)
            if s2 <= e2 and src[s2:e2] == t:
                return s2, e2, True
    lo = max(0, start - window)
    hi = min(n, end + window)
    idx = src.find(t, lo, hi)
    if idx != -1:
        return idx, idx + len(t), True
    return start, end, False


def normalize_spans(text: str, spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize spans. Extract and preserve span text (substrings).
    Offsets are optional and not required since we only store text in CSVs.
    """
    norm = []
    for sp in spans:
        lab = sp.get("label")
        if lab not in VALID_LABELS:
            continue
        tx = str(sp.get("text", "")).strip()
        if not tx:
            continue
        try:
            s = int(sp.get("start", -1))
            e = int(sp.get("end", -1))
        except Exception:
            s, e = -1, -1

        if s >= 0 and e > s and s < len(text) and e <= len(text) and text[s:e] == tx:
            norm.append({"label": lab, "start": s, "end": e, "text": tx})
        else:
            idx = text.find(tx)
            if idx >= 0:
                norm.append({"label": lab, "start": idx, "end": idx + len(tx), "text": tx})
            else:
                norm.append({"label": lab, "start": -1, "end": -1, "text": tx})
    return norm


def validate_result(res: Dict[str, Any], text: str, original_spans: List[Dict[str, Any]] | None = None) -> Tuple[bool, str]:
    spec = [
        ("is_conspiratorial", bool),
        ("rationale_short", str),
        ("confidence", (int, float)),
        ("spans", list),
    ]
    for k, t in spec:
        if k not in res:
            return False, f"Missing key: {k}"
        if not isinstance(res[k], t):
            return False, f"Bad type for {k}"
    if not res["is_conspiratorial"] and len(res["spans"]) > 0:
        return False, "No spans allowed when is_conspiratorial=false."
    if res["is_conspiratorial"]:
        spans_to_check = original_spans if original_spans is not None else res["spans"]
        has_core = any(sp.get("label") in {"plan_event", "secret"} for sp in spans_to_check)
        if not has_core:
            return False, "Conspiratorial requires at least one of plan_event or secret."
        for sp in res["spans"]:
            if not {"label", "text"} <= sp.keys():
                return False, "Span missing required fields (label, text)."
            if sp["label"] not in VALID_LABELS:
                return False, f"Invalid label: {sp['label']}"
            if not sp.get("text") or not str(sp.get("text")).strip():
                return False, "Span text cannot be empty."
    return True, ""


def _generate(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    input_ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(
        model.device
    )
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    eos_ids = [tokenizer.eos_token_id]
    try:
        eot_id = tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if eot_id not in (None, -1, tokenizer.eos_token_id):
            eos_ids.append(eot_id)
    except Exception:
        pass
    with torch.inference_mode():
        out = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=eos_ids,
            do_sample=False,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0], skip_special_tokens=False)


def generate_annotation(
    model,
    tokenizer,
    text: str,
    system_prompt: str,
    max_new_tokens: int = 512,
) -> Dict[str, Any]:
    messages = build_chat_prompt(system_prompt, text)
    decoded = _generate(model, tokenizer, messages, max_new_tokens)
    parsed = _last_top_level_json(decoded) or {}

    res = {
        "raw": decoded,
        "is_conspiratorial": parsed.get("is_conspiratorial"),
        "rationale_short": parsed.get("rationale_short"),
        "confidence": parsed.get("confidence"),
        "spans": parsed.get("spans", []),
    }

    original_spans = res["spans"].copy() if res["spans"] else []
    res["spans"] = normalize_spans(text, res.get("spans", []))
    ok, err = validate_result(res, text, original_spans=original_spans)
    if not ok:
        decoded2 = _generate(model, tokenizer, messages, max_new_tokens)
        parsed2 = _last_top_level_json(decoded2) or {}
        res2 = {
            "raw": decoded2,
            "is_conspiratorial": parsed2.get("is_conspiratorial"),
            "rationale_short": parsed2.get("rationale_short"),
            "confidence": parsed2.get("confidence"),
            "spans": parsed2.get("spans", []),
        }
        original_spans2 = res2["spans"].copy() if res2["spans"] else []
        res2["spans"] = normalize_spans(text, res2.get("spans", []))
        ok2, err2 = validate_result(res2, text, original_spans=original_spans2)
        if ok2:
            res = res2
            ok, err = ok2, err2
        else:
            if res2.get("is_conspiratorial") and err2.startswith("Conspiratorial requires at least one"):
                repair_user = (
                    "Return only JSON with a single 'spans' list containing at least one core span "
                    "(label 'plan_event' or 'secret'). Keep offsets exact and do not include quotes/punctuation.\n\n"
                    f"Input text\n{text}\n\nReturn only the JSON."
                )
                repair_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": repair_user},
                ]
                decoded3 = _generate(model, tokenizer, repair_messages, max_new_tokens=196)
                parsed3 = _last_top_level_json(decoded3) or {}
                spans3_raw = parsed3.get("spans", [])
                spans3 = normalize_spans(text, spans3_raw)
                if spans3:
                    res2["spans"] = spans3
                    ok3, err3 = validate_result(res2, text, original_spans=spans3_raw)
                    if ok3:
                        res = res2
                        ok, err = ok3, err3
            if not ok:
                res["validation_error"] = err
                return res
    res["validation_error"] = None
    return res


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_PATH, help="Few-shot prompt file (JSON-only prompt).")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Input CSV with a text column.")
    parser.add_argument("--out", default=DEFAULT_OUTPUT_PATH, help="Output JSONL path (optional, disabled by default).")
    parser.add_argument("--update_csv_out", default=DEFAULT_UPDATED_CSV, help="Merged CSV output path.")
    parser.add_argument("--fail_log", default=DEFAULT_FAIL_LOG, help="Failure diagnostics JSONL path.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)

    def parse_limit(value):
        if value is None or value == "None" or str(value).lower() == "none":
            return None
        return int(value)

    parser.add_argument(
        "--limit",
        type=parse_limit,
        default=DEFAULT_LIMIT,
        nargs="?",
        const=None,
        help="Limit number of unique texts to process (omit for all)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    output_prefix = get_output_prefix_from_prompt(args.prompt, args.model)
    dataset_suffix = slugify_filename(Path(args.csv).stem)

    if args.update_csv_out == DEFAULT_UPDATED_CSV:
        args.update_csv_out = f"outputs/{output_prefix}_{dataset_suffix}_annotations_raw.csv"
    if args.fail_log == DEFAULT_FAIL_LOG:
        args.fail_log = f"outputs/{output_prefix}_{dataset_suffix}_annotation_failures.jsonl"

    global LLM_ALL_CLASS_CSV, LLM_ALL_SPANS_CSV
    LLM_ALL_CLASS_CSV = f"outputs/{output_prefix}_{dataset_suffix}_all_classifications.csv"
    LLM_ALL_SPANS_CSV = f"outputs/{output_prefix}_{dataset_suffix}_all_span_annotations.csv"

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if args.update_csv_out:
        os.makedirs(os.path.dirname(args.update_csv_out), exist_ok=True)
    os.makedirs(os.path.dirname(args.fail_log), exist_ok=True)
    os.makedirs(os.path.dirname(LLM_ALL_CLASS_CSV), exist_ok=True)
    os.makedirs(os.path.dirname(LLM_ALL_SPANS_CSV), exist_ok=True)

    print(f"Using prompt: {args.prompt}")
    print(f"Output prefix: {output_prefix}")
    print(f"Classifications CSV: {LLM_ALL_CLASS_CSV}")
    print(f"Spans CSV: {LLM_ALL_SPANS_CSV}")

    system_prompt = load_prompt(args.prompt)
    human_df, samples, text_col = load_texts(args.csv, args.limit)

    text_id_col = "text_id" if "text_id" in human_df.columns else None

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_name_lower = args.model.lower()
    is_70b_model = "70b" in model_name_lower or "70-b" in model_name_lower

    if is_70b_model and BITSANDBYTES_AVAILABLE and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected 70B model. Using 4-bit quantization with {num_gpus} GPU(s)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=quantization_config,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto",
        )
    model.eval()

    print(f"Processing {len(samples)} samples with {args.model}...")
    results: List[Dict[str, Any]] = []
    fail_f = open(args.fail_log, "w", encoding="utf-8")
    try:
        for i, item in enumerate(samples):
            text = str(item["text"])[:20000]
            row_index = int(item["row_index"])
            text_id = item.get("text_id")
            if text_id is None:
                if text_id_col and row_index in human_df.index:
                    try:
                        text_id = human_df.loc[row_index, text_id_col]
                    except (KeyError, IndexError):
                        text_id = None
                if text_id is None:
                    text_id = row_index
            ann = generate_annotation(model, tokenizer, text, system_prompt, max_new_tokens=args.max_new_tokens)
            record = {
                "text_id": text_id,
                "row_index": row_index,
                "text": text,
                "llm_is_conspiratorial": ann.get("is_conspiratorial"),
                "llm_rationale_short": ann.get("rationale_short"),
                "llm_confidence": ann.get("confidence"),
                "llm_spans_json": json.dumps(ann.get("spans", []), ensure_ascii=False),
                "llm_raw": ann.get("raw"),
                "llm_validation_error": ann.get("validation_error"),
            }
            if record["llm_validation_error"]:
                fail_f.write(
                    json.dumps(
                        {
                            "row_index": row_index,
                            "reason": record["llm_validation_error"],
                            "raw_head": record["llm_raw"][:600],
                            "spans": json.loads(record["llm_spans_json"]),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            results.append(record)
            print(
                f"[{i + 1}/{len(samples)}] row={row_index} "
                f"is_consp={record['llm_is_conspiratorial']} "
                f"valid_err={record['llm_validation_error']}"
            )
    finally:
        fail_f.close()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fout:
            for rec in results:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote JSONL -> {args.out}")

    annot_col = None
    text_id_col = "text_id" if "text_id" in human_df.columns else None
    ann_id_col = "annotation_id" if "annotation_id" in human_df.columns else None

    def yn(v: Any) -> str:
        return "yes" if bool(v) else "no"

    class_rows: List[Dict[str, Any]] = []
    span_rows_per_text: List[Dict[str, Any]] = []

    meta_cols = [c for c in [text_id_col, ann_id_col, text_col] if c]
    meta_by_ridx = human_df.set_index("row_index")[meta_cols] if meta_cols else None

    annotation_counter = 0

    for rec in results:
        ridx = rec["row_index"]
        annotator_id = "llm"
        text_id = rec.get("text_id", ridx)
        annotation_counter += 1
        annotation_id = f"llm_{annotation_counter}"
        text_val = rec["text"]
        if meta_by_ridx is not None and ridx in meta_by_ridx.index:
            rowm = meta_by_ridx.loc[ridx]
            if text_col:
                text_val = rowm.get(text_col, text_val)
        class_rows.append(
            {
                "annotator_id": annotator_id,
                "text_id": text_id,
                "annotation_id": annotation_id,
                "text": text_val,
                "conspiracy": yn(rec["llm_is_conspiratorial"]),
                "supportive": "",
                "misinformative": "",
                "confidence": rec.get("llm_confidence"),
                "rationale_short": rec.get("llm_rationale_short"),
            }
        )
        label_to_texts: Dict[str, List[str]] = {lab: [] for lab in VALID_LABELS}
        spans = json.loads(rec.get("llm_spans_json", "[]")) if rec.get("llm_spans_json") else []
        for sp in spans:
            lab = sp.get("label")
            tx = sp.get("text")
            if lab in label_to_texts and isinstance(tx, str) and tx:
                label_to_texts[lab].append(tx)
        span_rows_per_text.append(
            {
                "annotator_id": annotator_id,
                "text_id": text_id,
                "annotation_id": annotation_id,
                "text": text_val,
                "out-group": " || ".join(label_to_texts["out_group"]) if label_to_texts["out_group"] else "",
                "plan": " || ".join(label_to_texts["plan_event"]) if label_to_texts["plan_event"] else "",
                "call-to-action": " || ".join(label_to_texts["call_to_action"]) if label_to_texts["call_to_action"] else "",
                "in-group": " || ".join(label_to_texts["in_group"]) if label_to_texts["in_group"] else "",
                "secret": " || ".join(label_to_texts["secret"]) if label_to_texts["secret"] else "",
            }
        )

    pd.DataFrame(class_rows).to_csv(LLM_ALL_CLASS_CSV, index=False)
    pd.DataFrame(span_rows_per_text).to_csv(LLM_ALL_SPANS_CSV, index=False)
    print(f"Wrote {LLM_ALL_CLASS_CSV} and {LLM_ALL_SPANS_CSV}")

    merged = pd.read_csv(args.csv)
    merged = merged.reset_index(drop=False).rename(columns={"index": "row_index"})
    res_df = pd.DataFrame(results)

    for col in [
        "llm_is_conspiratorial",
        "llm_rationale_short",
        "llm_confidence",
        "llm_spans_json",
        "llm_raw",
        "llm_validation_error",
    ]:
        if col not in merged.columns:
            merged[col] = None

    merged = merged.merge(
        res_df[
            [
                "row_index",
                "llm_is_conspiratorial",
                "llm_rationale_short",
                "llm_confidence",
                "llm_spans_json",
                "llm_raw",
                "llm_validation_error",
            ]
        ],
        on="row_index",
        how="left",
        suffixes=("", "_new"),
    )

    for col in [
        "llm_is_conspiratorial",
        "llm_rationale_short",
        "llm_confidence",
        "llm_spans_json",
        "llm_raw",
        "llm_validation_error",
    ]:
        new_col = f"{col}_new"
        if new_col in merged.columns:
            merged[col] = merged[new_col].combine_first(merged[col])
            merged.drop(columns=[new_col], inplace=True)

    merged.drop(columns=["row_index"], inplace=True, errors="ignore")
    merged.to_csv(args.update_csv_out, index=False)
    print(f"Updated CSV -> {args.update_csv_out}")
    print("Done.")


if __name__ == "__main__":
    main()


