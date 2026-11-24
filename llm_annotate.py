#!/usr/bin/env python3
"""
Annotate N samples using LLM models (Llama-3-8B-Instruct, Llama-3.3-70B-Instruct, etc.).

- Reads zero-shot prompt from Data/prompts/llama_zero_shot_prompt.txt
  (the prompt you provided; keep it EXACT)
- Reads texts from CSV (default Data/all_classifications.csv)
  and detects the text column among: text, content, body, post, article_text
- For each text: gets JSON with:
    is_conspiratorial (bool)
    supports_conspiracy (bool)
    rationale_short (str)
    confidence (float)
    spans (list of {label,text,start,end})  — only when conspiratorial
- Optionally writes JSONL if --out is provided (disabled by default)
- Writes a copy of the CSV with LLM columns merged in (by row index)
- Also writes normalized CSVs:
    Data/llm_classifications.csv (row_index,text,is_conspiratorial,supports_conspiracy,rationale_short,confidence)
    Data/llm_spans.csv           (row_index,label,start,end,span_text)
- And human-schema-mirrored CSVs:
    Data/llm_all_classifications.csv (annotator_id,text_id,annotation_id,text,conspiracy,supportive,misinformative,confidence,rationale_short)
    Data/llm_all_span_annotations.csv (annotator_id,text_id,annotation_id,text,plan_event,secret,out_group,in_group,call_to_action)
"""
import os
import re
import json
import argparse
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
DEFAULT_PROMPT_PATH = "Data/prompts/llama_zero_shot_prompt.txt"
DEFAULT_CSV_PATH = "Data/all_classifications.csv"
DEFAULT_OUTPUT_PATH = None  # JSONL output disabled by default
DEFAULT_UPDATED_CSV = "outputs/llama8b-zeroshot_all_classifications_raw.csv"
DEFAULT_FAIL_LOG = "outputs/llama8b-zeroshot_annotation_failures.jsonl"
DEFAULT_LIMIT = None  # None means process all unique texts
VALID_LABELS = {"plan_event", "secret", "out_group", "in_group", "call_to_action"}
PUNCT = "\"'""'`,.;:!?()[]{}"

# Output paths - will be adjusted based on prompt filename
LLM_CLASS_CSV = "outputs/llama8b-zeroshot_classifications.csv"  # deprecated (not written)
LLM_SPANS_CSV = "outputs/llama8b-zeroshot_spans.csv"            # deprecated (not written)
LLM_ALL_CLASS_CSV = "outputs/llama8b-zeroshot_all_classifications.csv"
LLM_ALL_SPANS_CSV = "outputs/llama8b-zeroshot_all_span_annotations.csv"


def get_output_prefix_from_prompt(prompt_path: str, model_name: str = None) -> str:
    """Detect prompt type from filename and model name, return appropriate output prefix."""
    prompt_name = os.path.basename(prompt_path).lower()
    
    # Detect model size from model name/path
    # Format: "llama-3.3-70B" or "llama8b" or "llama70b"
    model_prefix = "llama8b"  # default
    if model_name:
        model_name_lower = model_name.lower()
        # Check for Llama 3.3 70B
        if ("3.3" in model_name_lower and ("70" in model_name_lower or "70b" in model_name_lower)) or \
           ("70b" in model_name_lower or "70-b" in model_name_lower):
            model_prefix = "llama-3.3-70B"
        # Check for standard 70B (without 3.3)
        elif "70b" in model_name_lower and "3.3" not in model_name_lower:
            model_prefix = "llama70b"
        # Check for 8B models
        elif "8b" in model_name_lower or "8-b" in model_name_lower:
            model_prefix = "llama8b"
    
    # Detect prompt type
    prompt_type = "zeroshot"
    if "frame" in prompt_name:
        prompt_type = "frame"
    elif "fewshot" in prompt_name or "few_shot" in prompt_name:
        prompt_type = "fewshot"
    
    return f"{model_prefix}-{prompt_type}"


def load_prompt(prompt_path: str) -> str:
    with open(prompt_path, "r", encoding="utf-8") as f:
        return f.read().strip()


def detect_text_col(df: pd.DataFrame) -> str:
    for cand in ["text", "content", "body", "post", "article_text"]:
        if cand in df.columns:
            return cand
    raise ValueError(
        "No suitable text column found in CSV. Expected one of: "
        "text, content, body, post, article_text"
    )


def load_texts(csv_path: str, limit: int | None = DEFAULT_LIMIT) -> Tuple[pd.DataFrame, List[Dict[str, Any]], str]:
    """
    Load texts, but only unique text_ids (LLM is a single annotator, should annotate each text once).
    If limit is None, process all unique texts.
    """
    df = pd.read_csv(csv_path)
    text_col = detect_text_col(df)
    df = df.reset_index(drop=False).rename(columns={"index": "row_index"})
    df = df[df[text_col].astype(str).str.len() > 0]
    
    # For LLM annotation: only process unique text_ids (LLM is one annotator)
    # Keep first occurrence of each unique text_id
    if "text_id" in df.columns:
        df_unique = df.drop_duplicates(subset="text_id", keep="first")
        print(f"Filtered to unique text_ids: {len(df)} -> {len(df_unique)} rows")
        df = df_unique
    
    # Apply limit after deduplication (None means process all)
    if limit is None:
        subset = df[["row_index", text_col]].rename(columns={text_col: "text"})
        if "text_id" in df.columns:
            text_ids = df["text_id"].values
        else:
            text_ids = None
        print(f"Processing all {len(df)} unique texts (limit=None)")
    else:
        subset = df.head(limit)[["row_index", text_col]].rename(columns={text_col: "text"})
        if "text_id" in df.columns:
            text_ids = df.head(limit)["text_id"].values
        else:
            text_ids = None
        print(f"Processing {len(subset)} unique texts (limit={limit})")
    
    # Also keep text_id if available for later use
    samples = subset.to_dict(orient="records")
    if "text_id" in df.columns and text_ids is not None:
        for i, sample in enumerate(samples):
            sample["text_id"] = text_ids[i] if i < len(text_ids) else None
    
    return df, samples, text_col


def build_chat_prompt(system_prompt: str, text: str) -> List[Dict[str, str]]:
    # Handle prompts that use {{INPUT_TEXT}} or {INPUT_TEXT} placeholder
    if "{{INPUT_TEXT}}" in system_prompt or "{INPUT_TEXT}" in system_prompt:
        # Replace placeholder with actual text (prompt already has instructions)
        user_block = system_prompt.replace("{{INPUT_TEXT}}", text).replace("{INPUT_TEXT}", text)
        # If prompt has placeholder, use entire prompt as user message (few-shot style)
        return [{"role": "user", "content": user_block}]
    else:
        # Legacy zero-shot prompt: use system prompt + user message with text
        user_block = f"Input text\n{text}\n\nReturn only the JSON."
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_block},
        ]


def _last_top_level_json(s: str) -> Dict:
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
            return json.loads(s[first:last + 1])
        except Exception:
            continue
    m = list(re.finditer(r"\{[\s\S]*?\}", s))
    if m:
        for mm in reversed(m):
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


def normalize_spans(text: str, spans: List[Dict]) -> List[Dict]:
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
        # Try to get offsets if provided (optional), otherwise find text in document
        try:
            s = int(sp.get("start", -1)); e = int(sp.get("end", -1))
        except Exception:
            s, e = -1, -1
        
        # If offsets provided and match, use them; otherwise find text in document
        if s >= 0 and e > s and s < len(text) and e <= len(text) and text[s:e] == tx:
            # Offsets are correct
            norm.append({"label": lab, "start": s, "end": e, "text": tx})
        else:
            # Try to find text in document for approximate offsets (optional)
            idx = text.find(tx)
            if idx >= 0:
                norm.append({"label": lab, "start": idx, "end": idx + len(tx), "text": tx})
            else:
                # Keep span text without offsets (offsets not stored in CSVs anyway)
                norm.append({"label": lab, "start": -1, "end": -1, "text": tx})
    return norm


def validate_result(res: Dict, text: str, original_spans: List[Dict] = None) -> Tuple[bool, str]:
    """
    Validate result. original_spans can be provided to check span presence
    even if normalized spans were removed due to offset mismatches.
    """
    # Handle null/None for supports_conspiracy (convert to False)
    if res.get("supports_conspiracy") is None:
        res["supports_conspiracy"] = False
    
    # Required fields
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
    
    # supports_conspiracy is optional (defaults to False if not provided)
    # Only validate if present
    if "supports_conspiracy" in res:
        if not isinstance(res["supports_conspiracy"], bool):
            return False, f"Bad type for supports_conspiracy"
        if res["is_conspiratorial"] and res["supports_conspiracy"]:
            return False, "Mutual exclusivity violated (both true)."
    if not res["is_conspiratorial"] and len(res["spans"]) > 0:
        return False, "No spans allowed when is_conspiratorial=false."
    if res["is_conspiratorial"]:
        # Check for core spans - use original_spans if provided (before normalization),
        # or normalized spans. Since offsets aren't stored, we just need span presence.
        spans_to_check = original_spans if original_spans is not None else res["spans"]
        has_core = any(sp.get("label") in {"plan_event", "secret"} for sp in spans_to_check)
        if not has_core:
            return False, "Conspiratorial requires at least one of plan_event or secret."
        # Validate span structure - only require label and text (offsets optional)
        for sp in res["spans"]:
            if not {"label", "text"} <= sp.keys():
                return False, "Span missing required fields (label, text)."
            if sp["label"] not in VALID_LABELS:
                return False, f"Invalid label: {sp['label']}"
            if not sp.get("text") or not str(sp.get("text")).strip():
                return False, "Span text cannot be empty."
            # Note: start/end offsets are optional - not required since we only store text
    return True, ""


def _generate(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int,
) -> str:
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)
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
    max_new_tokens: int = 512
) -> Dict[str, Any]:
    messages = build_chat_prompt(system_prompt, text)
    decoded = _generate(model, tokenizer, messages, max_new_tokens)
    parsed = _last_top_level_json(decoded) or {}
    # Handle null/None for supports_conspiracy (convert to False)
    supports_val = parsed.get("supports_conspiracy")
    if supports_val is None:
        supports_val = False
    
    res = {
        "raw": decoded,
        "is_conspiratorial": parsed.get("is_conspiratorial"),
        "supports_conspiracy": supports_val,
        "rationale_short": parsed.get("rationale_short"),
        "confidence": parsed.get("confidence"),
        "spans": parsed.get("spans", []),
    }
    # Save original spans before normalization (for validation)
    original_spans = res["spans"].copy() if res["spans"] else []
    res["spans"] = normalize_spans(text, res.get("spans", []))
    # Validate using original_spans to check for core span presence
    ok, err = validate_result(res, text, original_spans=original_spans)
    if not ok:
        decoded2 = _generate(model, tokenizer, messages, max_new_tokens)
        parsed2 = _last_top_level_json(decoded2) or {}
        # Handle null/None for supports_conspiracy
        supports_val2 = parsed2.get("supports_conspiracy")
        if supports_val2 is None:
            supports_val2 = False
        res2 = {
            "raw": decoded2,
            "is_conspiratorial": parsed2.get("is_conspiratorial"),
            "supports_conspiracy": supports_val2,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default=DEFAULT_PROMPT_PATH, help="Zero-shot prompt file (your JSON-only prompt).")
    parser.add_argument("--csv", default=DEFAULT_CSV_PATH, help="Input CSV with a text column.")
    parser.add_argument("--out", default=None, help="Output JSONL path (optional, disabled by default).")
    parser.add_argument("--update_csv_out", default=DEFAULT_UPDATED_CSV, help="Merged CSV output path.")
    parser.add_argument("--fail_log", default=DEFAULT_FAIL_LOG, help="Failure diagnostics JSONL path.")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID)
    def parse_limit(value):
        """Parse limit argument: None means all, otherwise convert to int."""
        if value is None or value == 'None' or str(value).lower() == 'none':
            return None
        return int(value)
    
    parser.add_argument("--limit", type=parse_limit, default=DEFAULT_LIMIT, nargs='?', const=None, help="Limit number of unique texts to process (omit for all)")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    # Determine output prefix based on prompt filename and model name
    output_prefix = get_output_prefix_from_prompt(args.prompt, args.model)
    
    # Update output paths based on detected prefix
    if args.update_csv_out == DEFAULT_UPDATED_CSV:
        args.update_csv_out = f"outputs/{output_prefix}_all_classifications_raw.csv"
    if args.fail_log == DEFAULT_FAIL_LOG:
        args.fail_log = f"outputs/{output_prefix}_annotation_failures.jsonl"
    
    # Update global CSV paths
    global LLM_ALL_CLASS_CSV, LLM_ALL_SPANS_CSV
    LLM_ALL_CLASS_CSV = f"outputs/{output_prefix}_all_classifications.csv"
    LLM_ALL_SPANS_CSV = f"outputs/{output_prefix}_all_span_annotations.csv"

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
    
    # Define text_id_col early for use in annotation loop
    text_id_col = "text_id" if "text_id" in human_df.columns else None

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    
    # For 70B models, use 4-bit quantization to fit across multiple GPUs
    # Check if model is 70B by checking path/name
    model_name_lower = args.model.lower()
    is_70b_model = "70b" in model_name_lower or "70-b" in model_name_lower
    
    if is_70b_model and BITSANDBYTES_AVAILABLE and torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"Detected 70B model. Using 4-bit quantization with {num_gpus} GPU(s)...")
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
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
            text_id = item.get("text_id")  # Get text_id from sample (should be set by load_texts)
            if text_id is None:
                # Fallback: try to get from human_df using row_index
                if text_id_col and row_index in human_df.index:
                    try:
                        text_id = human_df.loc[row_index, text_id_col]
                    except (KeyError, IndexError):
                        text_id = None
                if text_id is None:
                    text_id = row_index  # Last resort: use row_index
            ann = generate_annotation(model, tokenizer, text, system_prompt, max_new_tokens=args.max_new_tokens)
            record = {
                "text_id": text_id,  # Store text_id in result
                "row_index": row_index,
                "text": text,
                "llm_is_conspiratorial": ann.get("is_conspiratorial"),
                "llm_supports_conspiracy": ann.get("supports_conspiracy"),
                "llm_rationale_short": ann.get("rationale_short"),
                "llm_confidence": ann.get("confidence"),
                "llm_spans_json": json.dumps(ann.get("spans", []), ensure_ascii=False),
                "llm_raw": ann.get("raw"),
                "llm_validation_error": ann.get("validation_error"),
            }
            if record["llm_validation_error"]:
                fail_f.write(json.dumps({
                    "row_index": row_index,
                    "reason": record["llm_validation_error"],
                    "raw_head": record["llm_raw"][:600],
                    "spans": json.loads(record["llm_spans_json"]),
                }, ensure_ascii=False) + "\n")
            results.append(record)
            print(
                f"[{i+1}/{len(samples)}] row={row_index} "
                f"is_consp={record['llm_is_conspiratorial']} "
                f"supports={record['llm_supports_conspiracy']} "
                f"valid_err={record['llm_validation_error']}"
            )
    finally:
        fail_f.close()

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fout:
            for rec in results:
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"Wrote JSONL -> {args.out}")

    # (Normalized CSVs skipped; using schema-mirrored CSVs instead)

    # Emit human-schema-mirrored CSVs
    # Map row_index -> (annotator_id,text_id,annotation_id,text) when available
    # Always tag LLM outputs with annotator_id="llm"; do not inherit human annotator ids
    annot_col = None
    text_id_col = "text_id" if "text_id" in human_df.columns else None
    ann_id_col = "annotation_id" if "annotation_id" in human_df.columns else None

    def yn(v: Any) -> str:
        return "yes" if bool(v) else "no"

    # llm_all_classifications.csv
    class_rows = []
    # llm_all_span_annotations.csv (one row per text with per-label substrings joined by ' || ')
    span_rows_per_text = []

    # Build fast lookup for original meta by row_index
    meta_cols = [c for c in [text_id_col, ann_id_col, text_col] if c]
    meta_by_ridx = human_df.set_index("row_index")[meta_cols] if meta_cols else None

    # Use incremental annotation_id (llm_1, llm_2, ...) instead of row_index
    annotation_counter = 0
    
    for rec in results:
        ridx = rec["row_index"]
        # meta defaults
        annotator_id = "llm"
        # Use text_id from result (set during annotation) or fallback
        text_id = rec.get("text_id", ridx)  # Prefer text_id from annotation result
        annotation_counter += 1
        annotation_id = f"llm_{annotation_counter}"  # Incremental: llm_1, llm_2, ...
        text_val = rec["text"]
        # Copy text from input CSV if available (text_id already set)
        if meta_by_ridx is not None and ridx in meta_by_ridx.index:
            rowm = meta_by_ridx.loc[ridx]
            if text_col: text_val = rowm.get(text_col, text_val)  # Copy text from input
            # Note: annotation_id is always incremental, not inherited from input
            # Note: text_id is preserved from annotation result
        # classifications row
        class_rows.append({
            "annotator_id": annotator_id,
            "text_id": text_id,
            "annotation_id": annotation_id,
            "text": text_val,
            "conspiracy": yn(rec["llm_is_conspiratorial"]),
            "supportive": yn(rec["llm_supports_conspiracy"]),
            "misinformative": "",  # not produced by this run
            "confidence": rec.get("llm_confidence"),
            "rationale_short": rec.get("llm_rationale_short"),
        })
        # spans row (per-label columns with joined substrings)
        label_to_texts: Dict[str, List[str]] = {lab: [] for lab in VALID_LABELS}
        spans = json.loads(rec.get("llm_spans_json", "[]")) if rec.get("llm_spans_json") else []
        for sp in spans:
            lab = sp.get("label")
            tx = sp.get("text")
            if lab in label_to_texts and isinstance(tx, str) and tx:
                label_to_texts[lab].append(tx)
        # Match human CSV column names and order exactly
        span_rows_per_text.append({
            "annotator_id": annotator_id,
            "text_id": text_id,
            "annotation_id": annotation_id,
            "text": text_val,
            "out-group": " || ".join(label_to_texts["out_group"]) if label_to_texts["out_group"] else "",  # Match human CSV column name and order
            "plan": " || ".join(label_to_texts["plan_event"]) if label_to_texts["plan_event"] else "",  # Match human CSV column name and order
            "call-to-action": " || ".join(label_to_texts["call_to_action"]) if label_to_texts["call_to_action"] else "",  # Match human CSV column name and order
            "in-group": " || ".join(label_to_texts["in_group"]) if label_to_texts["in_group"] else "",  # Match human CSV column name and order
            "secret": " || ".join(label_to_texts["secret"]) if label_to_texts["secret"] else "",
        })

    pd.DataFrame(class_rows).to_csv(LLM_ALL_CLASS_CSV, index=False)
    pd.DataFrame(span_rows_per_text).to_csv(LLM_ALL_SPANS_CSV, index=False)
    print(f"Wrote {LLM_ALL_CLASS_CSV} and {LLM_ALL_SPANS_CSV}")

    merged = pd.read_csv(args.csv)
    merged = merged.reset_index(drop=False).rename(columns={"index": "row_index"})
    res_df = pd.DataFrame(results)

    for col in [
        "llm_is_conspiratorial",
        "llm_supports_conspiracy",
        "llm_rationale_short",
        "llm_confidence",
        "llm_spans_json",
        "llm_raw",
        "llm_validation_error",
    ]:
        if col not in merged.columns:
            merged[col] = None

    merged = merged.merge(
        res_df[["row_index",
                "llm_is_conspiratorial",
                "llm_supports_conspiracy",
                "llm_rationale_short",
                "llm_confidence",
                "llm_spans_json",
                "llm_raw",
                "llm_validation_error"]],
        on="row_index",
        how="left",
        suffixes=("", "_new"),
    )

    for col in [
        "llm_is_conspiratorial",
        "llm_supports_conspiracy",
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
