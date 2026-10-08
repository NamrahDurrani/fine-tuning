#!/usr/bin/env python
"""
AgriBot - Qwen3 5,000-example baseline experiment (one question only):
"Does LoRA fine-tuning on the 5,000 agriculture examples improve MCQ accuracy on the frozen
100-question benchmark, compared with the untouched Qwen3 base model?"

Run from the repository root (or set AGRIBOT_REPO):
    python agribot_qwen3_5000_baseline.py --dry-run     # everything except GPU work
    python agribot_qwen3_5000_baseline.py               # full experiment

Order of operations (per the experiment spec):
 1 inspect repo -> 2 dataset stats -> 3 benchmark validation -> 4 leakage check (stops on leakage)
 -> 5 deterministic 90/10 split -> 6 BASE benchmark (before any training) -> 7 print config ->
 8 QLoRA/LoRA training -> 9 FINE-TUNED benchmark (adapter reloaded from disk) -> 10 report.

Nothing here edits the dataset or the benchmark. Nothing is invented: unparseable model outputs
are counted as incorrect and reported separately.
"""
import os, sys, re, json, time, math, random, hashlib, inspect, gc, argparse
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter, defaultdict

# ----------------------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------------------
REPO_ROOT = Path(os.environ.get("AGRIBOT_REPO", Path(__file__).resolve().parent))
DATASET_NAME = "AgriBot_5000_Unique_Agriculture_Dataset.json"
BENCHMARK_NAME = "crop_100_questions.json"
MODEL_NAME = "Qwen/Qwen3-4B-Thinking-2507"
EXP_NAME = "qwen3_5000_baseline"
EXP_DIR = REPO_ROOT / "AgriBot_Project" / "weights_qwen3_5000_baseline"
FINAL_DIR = EXP_DIR / "final_adapter"
EVAL_DIR = EXP_DIR / "evaluation"
REPORT_PATH = REPO_ROOT / "evaluation_reports" / "qwen3_5000_baseline_report.md"

SYSTEM_PROMPT = ("You are AgriBot, an agricultural assistant. Give accurate, practical, "
                 "safe advice and defer to local extension guidance for product-specific decisions.")

# Hyperparameters. "source" says where each value comes from. The values marked
# repo:agribot_nb1_local.py are the ones recorded in this repo's existing Qwen3 training script;
# the ones marked SPEC come from the experiment spec; NEW means nothing in the repo records it.
HP = {
    "lora_r": (16, "SPEC"), "lora_alpha": (32, "SPEC"), "lora_dropout": (0.05, "SPEC"),
    "target_modules": (["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"], "SPEC"),
    "use_dora": (False, "SPEC (baseline: no DoRA)"),
    "max_seq_length": (2560, "SPEC"), "packing": (False, "SPEC"),
    "learning_rate": (2e-4, "repo:agribot_nb1_local.py"),
    "num_train_epochs": (3, "repo:agribot_nb1_local.py"),
    "per_device_train_batch_size": (2, "repo:agribot_nb1_local.py"),
    "gradient_accumulation_steps": (8, "repo:agribot_nb1_local.py"),
    "lr_scheduler_type": ("cosine", "repo:agribot_nb1_local.py"),
    "warmup_ratio": (0.03, "repo:agribot_nb1_local.py"),
    "weight_decay": (0.01, "repo:agribot_nb1_local.py"),
    "max_grad_norm": (0.3, "repo:agribot_nb1_local.py"),
    "optim": ("paged_adamw_8bit", "repo:agribot_nb1_local.py"),
    "gradient_checkpointing": (True, "repo:agribot_nb1_local.py"),
    "eval_steps": (100, "repo:agribot_nb1_local.py"),
    "save_steps": (500, "repo:agribot_nb1_local.py"),
    "val_fraction": (0.10, "SPEC"),
}
SEED_DEFAULT = 42
GEN = {"do_sample": False, "temperature": None, "top_p": None, "max_new_tokens": 64,
       "repetition_penalty": 1.0}
NEAR_DUP_REPORT_COS = 0.90      # reported as "suspicious"
NEAR_DUP_STOP_COS = 0.97        # stops the run (configurable; exact/normalized overlap always stops)
ROMAN = ["I", "II", "III", "IV", "V", "VI"]
LABEL_MAP = {"I": 0, "II": 1, "III": 2, "IV": 3, "V": 4, "VI": 5,
             "A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5}
LABEL_ALT = "|".join(sorted(LABEL_MAP, key=len, reverse=True))

# ----------------------------------------------------------------------------------------
# small pure helpers (no heavy imports - safe to import/test anywhere)
# ----------------------------------------------------------------------------------------
def norm_text(s):
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()

def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)

def find_best_list(obj, need_any):
    best = []
    def walk(n):
        nonlocal best
        if isinstance(n, list) and n and isinstance(n[0], dict):
            keys = {k.lower() for k in n[0].keys()}
            if keys & need_any and len(n) > len(best):
                best = n
        if isinstance(n, dict):
            for v in n.values():
                walk(v)
        elif isinstance(n, list):
            for it in n:
                if isinstance(it, (dict, list)):
                    walk(it)
    walk(obj)
    return best

def first_val(d, keys):
    low = {k.lower(): k for k in d}
    for k in keys:
        if k in low and d[low[k]] not in (None, ""):
            return d[low[k]]
    return None

def extract_choice(text, n_options):
    """Maps a model's reply to an option index, or None if it cannot be parsed robustly."""
    t = (text or "").strip()
    pats = [
        r"^\s*[\(\[]?(%s)(?:[\)\]\.:,]|\s*$|\s*\n)" % LABEL_ALT,
        r"(?:[Aa]nswer|[Oo]ption|[Cc]hoice)\s*(?:is|:|=)?\s*[\(\[]?(%s)(?:[\)\]\.:,\s]|$)" % LABEL_ALT,
    ]
    for p in pats:
        m = re.search(p, t)
        if m:
            idx = LABEL_MAP[m.group(1)]
            return idx if idx < n_options else None
    hits = re.findall(r"[\(\[](%s)[\)\]]" % LABEL_ALT, t)
    if hits:
        idx = LABEL_MAP[hits[-1]]
        return idx if idx < n_options else None
    return None

def mcnemar_exact(b_only, c_only):
    n = b_only + c_only
    if n == 0:
        return 1.0
    k = min(b_only, c_only)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)

def paired_bootstrap_delta(a_flags, b_flags, n_boot=10000, seed=42):
    import numpy as np
    rng = np.random.default_rng(seed)
    d = np.array(b_flags, dtype=float) - np.array(a_flags, dtype=float)
    boots = [rng.choice(d, size=len(d), replace=True).mean() for _ in range(n_boot)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)

def sha256_of(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def free():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

def set_seeds(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np
        np.random.seed(seed)
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

def die(msg, code=2):
    print("\n" + "=" * 78 + "\nSTOP: " + msg + "\n" + "=" * 78)
    sys.exit(code)

# ----------------------------------------------------------------------------------------
# 1. repository inspection
# ----------------------------------------------------------------------------------------
def inspect_repo(root):
    skip = {".git", "__pycache__", ".venv", "venv", "node_modules"}
    files = [p for p in root.rglob("*") if p.is_file() and not any(s in p.parts for s in skip)]
    rel = lambda p: str(p.relative_to(root))
    info = {
        "root": str(root), "n_files": len(files),
        "python_scripts": sorted(rel(p) for p in files if p.suffix == ".py"),
        "notebooks": sorted(rel(p) for p in files if p.suffix == ".ipynb"),
        "json_files": sorted(rel(p) for p in files if p.suffix in (".json", ".jsonl")
                             and "checkpoint" not in p.parts),
        "requirements": sorted(rel(p) for p in files if p.name.startswith("requirements")),
        "existing_adapters": sorted(rel(p.parent) for p in files if p.name == "adapter_config.json"),
        "existing_reports": sorted(rel(p) for p in files if "evaluation_reports" in p.parts),
    }
    # informational scan of recorded hyperparameters / model ids in existing scripts
    pats = {"learning_rate": r"learning_rate\s*=\s*([0-9eE\.\-]+)",
            "epochs": r"num_train_epochs\s*=\s*(\d+)",
            "batch": r"per_device_train_batch_size\s*=\s*(\d+)",
            "grad_accum": r"gradient_accumulation_steps\s*=\s*(\d+)",
            "seed": r"\bSEED\s*=\s*(\d+)",
            "model_ids": r"[\"'](Qwen/Qwen[\w\.\-]+)[\"']"}
    recorded, model_ids = defaultdict(list), set()
    for s in info["python_scripts"]:
        try:
            txt = (root / s).read_text(errors="ignore")
        except Exception:
            continue
        for k, p in pats.items():
            for m in re.findall(p, txt):
                if k == "model_ids":
                    model_ids.add(m)
                else:
                    recorded[k].append((s, m))
    info["recorded_hyperparameters"] = {k: v[:6] for k, v in recorded.items()}
    info["model_ids_in_repo"] = sorted(model_ids)
    return info

# ----------------------------------------------------------------------------------------
# 2-3. dataset and benchmark
# ----------------------------------------------------------------------------------------
def load_dataset_rows(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    rows = find_best_list(obj, {"instruction", "question", "prompt"})
    if not rows:
        die("No instruction/response-shaped list found in " + str(path))
    return rows

def dataset_statistics(rows, max_len, tokenizer=None):
    st = {"n_examples": len(rows)}
    st["fields"] = sorted({k for r in rows if isinstance(r, dict) for k in r})
    malformed, empty = [], []
    qs, ans, cats = [], [], Counter()
    for i, r in enumerate(rows):
        if not isinstance(r, dict):
            malformed.append(i)
            continue
        q = first_val(r, ["instruction", "question", "prompt"])
        a = first_val(r, ["response", "output", "answer"])
        if q is None or a is None:
            empty.append(i)
            continue
        qs.append(str(q).strip()); ans.append(str(a).strip())
        cats[str(r.get("category", "Unknown"))] += 1
    st["malformed_records"] = len(malformed)
    st["empty_or_missing_question_or_answer"] = len(empty)
    st["duplicate_questions_exact"] = len(qs) - len(set(qs))
    st["duplicate_questions_normalized"] = len(qs) - len({norm_text(q) for q in qs})
    st["duplicate_answers_exact"] = len(ans) - len(set(ans))
    st["category_distribution"] = dict(cats)
    lens = sorted(len(q) + len(a) for q, a in zip(qs, ans))
    st["chars_min_median_max"] = [lens[0], lens[len(lens) // 2], lens[-1]] if lens else []
    if tokenizer is not None and qs:
        toks = [len(tokenizer(q + " " + a)["input_ids"]) for q, a in zip(qs, ans)]
        st["tokens_max"] = max(toks)
        st["examples_over_max_seq_length"] = sum(1 for t in toks if t > max_len)
    return st, malformed, empty

def load_benchmark(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = raw if isinstance(raw, list) else find_best_list(raw, {"question"})
    if len(rows) != 100:
        die("Benchmark must contain exactly 100 questions, found %d." % len(rows))
    problems, bench = [], []
    for i, q in enumerate(rows):
        opts = q.get("options")
        if not isinstance(opts, list) or len(opts) < 2:
            problems.append(q.get("question_id", q.get("id", i)))
            continue
        gold = None
        if q.get("answer_roman") in ROMAN:
            gold = ROMAN.index(q["answer_roman"])
        if isinstance(q.get("answer_index"), int):
            if gold is not None and gold != q["answer_index"]:
                die("Benchmark question %s: answer_index (%s) disagrees with answer_roman (%s). "
                    "Not guessing which is right - fix the file's metadata and re-run."
                    % (q.get("question_id"), q["answer_index"], q["answer_roman"]))
            gold = q["answer_index"]
        if gold is None or gold >= len(opts):
            problems.append(q.get("question_id", i))
            continue
        bench.append({"question_id": q.get("question_id", q.get("id", "Q%03d" % (i + 1))),
                      "question": q["question"], "options": opts, "gold_index": gold,
                      "category": str(q.get("category", "Unknown")), "level": str(q.get("level", "Unknown"))})
    if problems:
        die("This benchmark file is not a multiple-choice benchmark the spec can score exactly.\n"
            "%d of 100 questions lack usable `options` + `answer_index`/`answer_roman` "
            "(e.g. %s).\nThe spec forbids scoring MCQ by free-text similarity and forbids editing "
            "the benchmark, so the run stops here. If this is the free-text crop_100_questions.json "
            "(gold_answer/gold_facts, no options), point BENCHMARK_NAME at the MCQ benchmark file "
            "instead." % (len(problems), problems[:5]))
    return bench

# ----------------------------------------------------------------------------------------
# 4. leakage check
# ----------------------------------------------------------------------------------------
def leakage_check(bench, rows):
    train_q = [str(first_val(r, ["instruction", "question", "prompt"]) or "") for r in rows]
    exact = {q.strip() for q in train_q}
    normd = {norm_text(q) for q in train_q}
    ex_hits = [b["question_id"] for b in bench if b["question"].strip() in exact]
    nm_hits = [b["question_id"] for b in bench if norm_text(b["question"]) in normd]
    from sentence_transformers import SentenceTransformer, util
    emb = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    be = emb.encode([b["question"] for b in bench], convert_to_tensor=True, batch_size=64)
    te = emb.encode(train_q, convert_to_tensor=True, batch_size=128, show_progress_bar=True)
    sim = util.cos_sim(be, te)
    mx, arg = sim.max(dim=1)
    flagged = []
    for i, b in enumerate(bench):
        s = float(mx[i])
        if s >= NEAR_DUP_REPORT_COS:
            flagged.append({"question_id": b["question_id"], "similarity": round(s, 4),
                            "benchmark_question": b["question"],
                            "nearest_training_question": train_q[int(arg[i])]})
    rep = {"benchmark_size": len(bench), "training_size": len(rows),
           "exact_overlap_count": len(ex_hits), "exact_overlap_ids": ex_hits,
           "normalized_overlap_count": len(nm_hits), "normalized_overlap_ids": nm_hits,
           "near_duplicate_count_ge_%.2f" % NEAR_DUP_REPORT_COS: len(flagged),
           "near_duplicate_count_ge_%.2f" % NEAR_DUP_STOP_COS: sum(1 for f in flagged if f["similarity"] >= NEAR_DUP_STOP_COS),
           "pct_benchmark_potentially_represented": round(100.0 * len(flagged) / len(bench), 1),
           "flagged": flagged}
    return rep

# ----------------------------------------------------------------------------------------
# prompt format (identical for base and fine-tuned; identical to the training format)
# ----------------------------------------------------------------------------------------
def labels_for(n):
    return ROMAN[:n]

def mcq_user_text(q):
    labs = labels_for(len(q["options"]))
    lines = [q["question"].strip(), "", "Options:"]
    for lab, opt in zip(labs, q["options"]):
        lines.append("(%s) %s" % (lab, opt))
    lines += ["", "Select the single best option. Reply with only the option label (%s) and nothing else."
              % ", ".join(labs)]
    return "\n".join(lines)

def chat_prompt(tokenizer, user_text):
    p = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_text}],
        tokenize=False, add_generation_prompt=True)
    if not p.rstrip().endswith("<think>"):
        p += "<think>\n"
    return p

# ----------------------------------------------------------------------------------------
# model loading / evaluation
# ----------------------------------------------------------------------------------------
def bnb_config():
    import torch
    from transformers import BitsAndBytesConfig
    dt = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                              bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=dt), dt

def load_for_eval(adapter_dir=None):
    import torch
    from transformers import AutoModelForCausalLM
    cfg, dt = bnb_config()
    m = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=cfg, device_map={"": 0},
                                             torch_dtype=dt, attn_implementation="sdpa",
                                             trust_remote_code=True)
    if adapter_dir:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(adapter_dir))
    m.eval()
    m.config.use_cache = True
    return m

def evaluate(label, bench, tokenizer, adapter_dir=None):
    import torch
    out_path = EVAL_DIR / (label + "_predictions.json")
    done = {}
    if out_path.exists():
        done = {r["question_id"]: r for r in json.loads(out_path.read_text())}
    if len(done) == len(bench):
        print("[%s] predictions already complete - reusing %s" % (label, out_path))
        return [done[b["question_id"]] for b in bench]
    model = load_for_eval(adapter_dir)
    for i, b in enumerate(bench):
        if b["question_id"] in done:
            continue
        prompt = chat_prompt(tokenizer, mcq_user_text(b)) + "\n</think>\n\n"   # empty think block, same for both models
        x = tokenizer(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            o = model.generate(**x, max_new_tokens=GEN["max_new_tokens"], do_sample=False,
                               temperature=None, top_p=None, repetition_penalty=GEN["repetition_penalty"],
                               pad_token_id=tokenizer.pad_token_id)
        raw = tokenizer.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        pred = extract_choice(raw, len(b["options"]))
        done[b["question_id"]] = {
            "question_id": b["question_id"], "category": b["category"], "level": b["level"],
            "gold_index": b["gold_index"], "gold_label": labels_for(len(b["options"]))[b["gold_index"]],
            "raw_output": raw, "pred_index": pred,
            "pred_label": labels_for(len(b["options"]))[pred] if pred is not None else None,
            "parsed": pred is not None, "correct": pred == b["gold_index"]}
        out_path.write_text(json.dumps(list(done.values()), indent=2))
        if (i + 1) % 10 == 0:
            print("  [%s] %d/%d" % (label, i + 1, len(bench)))
    del model
    free()
    return [done[b["question_id"]] for b in bench]

# ----------------------------------------------------------------------------------------
# training
# ----------------------------------------------------------------------------------------
def train(train_rows, val_rows, tokenizer, seed, hw):
    import torch
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import Dataset

    def to_pc(r):
        ins = str(first_val(r, ["instruction", "question", "prompt"])).strip()
        extra = str(r.get("input", "") or "").strip()
        if extra:
            ins = ins + "\n\n" + extra
        resp = str(first_val(r, ["response", "output", "answer"])).strip()
        return {"prompt": chat_prompt(tokenizer, ins), "completion": "\n</think>\n\n" + resp + "<|im_end|>\n"}

    tr_ds = Dataset.from_list([to_pc(r) for r in train_rows])
    va_ds = Dataset.from_list([to_pc(r) for r in val_rows])

    cfg, dt = bnb_config()
    use_bf16 = dt == torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=cfg, device_map={"": 0},
                                                 torch_dtype=dt, attn_implementation="sdpa",
                                                 trust_remote_code=True)
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = LoraConfig(r=HP["lora_r"][0], lora_alpha=HP["lora_alpha"][0],
                             lora_dropout=HP["lora_dropout"][0], bias="none", task_type="CAUSAL_LM",
                             target_modules=HP["target_modules"][0], use_dora=HP["use_dora"][0])

    kw = dict(output_dir=str(EXP_DIR / "checkpoints"), num_train_epochs=HP["num_train_epochs"][0],
              per_device_train_batch_size=HP["per_device_train_batch_size"][0],
              per_device_eval_batch_size=HP["per_device_train_batch_size"][0],
              gradient_accumulation_steps=HP["gradient_accumulation_steps"][0],
              learning_rate=HP["learning_rate"][0], lr_scheduler_type=HP["lr_scheduler_type"][0],
              weight_decay=HP["weight_decay"][0], max_grad_norm=HP["max_grad_norm"][0],
              optim=HP["optim"][0], fp16=not use_bf16, bf16=use_bf16,
              gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
              logging_steps=10, eval_steps=HP["eval_steps"][0], save_steps=HP["save_steps"][0],
              save_total_limit=3, report_to="none", seed=seed, data_seed=seed,
              completion_only_loss=True, packing=HP["packing"][0], dataset_num_proc=2)
    params = inspect.signature(SFTConfig.__init__).parameters
    if "warmup_ratio" in params:
        kw["warmup_ratio"] = HP["warmup_ratio"][0]
    else:
        kw["warmup_steps"] = HP["warmup_ratio"][0]
    kw["eval_strategy" if "eval_strategy" in params else "evaluation_strategy"] = "steps"
    kw["max_length" if "max_length" in params else "max_seq_length"] = HP["max_seq_length"][0]
    dropped = [k for k in kw if k not in params]
    if dropped:
        print("NOTE - SFTConfig in this trl version does not accept: %s (skipped)" % dropped)
    kw = {k: v for k, v in kw.items() if k in params}
    args = SFTConfig(**kw)

    tk = dict(model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds, peft_config=peft_config)
    if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
        tk["processing_class"] = tokenizer
    else:
        tk["tokenizer"] = tokenizer
    trainer = SFTTrainer(**tk)
    trainable, total = 0, 0
    for p in trainer.model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()

    ckpts = sorted((EXP_DIR / "checkpoints").glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])) \
        if (EXP_DIR / "checkpoints").exists() else []
    resume = str(ckpts[-1]) if ckpts else None
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume)
    elapsed = time.time() - t0

    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    trainer.model.save_pretrained(str(FINAL_DIR))
    tokenizer.save_pretrained(str(FINAL_DIR))
    log = trainer.state.log_history
    (EXP_DIR / "training_log.json").write_text(json.dumps(log, indent=2))
    tr = [l for l in log if "loss" in l]
    ev = [l for l in log if "eval_loss" in l]
    acc = [l["mean_token_accuracy"] for l in tr if "mean_token_accuracy" in l]
    metrics = {"initial_train_loss": tr[0]["loss"] if tr else None,
               "final_train_loss": tr[-1]["loss"] if tr else None,
               "best_val_loss": min(l["eval_loss"] for l in ev) if ev else None,
               "final_val_loss": ev[-1]["eval_loss"] if ev else None,
               "final_mean_token_accuracy": acc[-1] if acc else None,
               "training_steps": trainer.state.global_step, "wall_clock_seconds": round(elapsed, 1),
               "trainable_params": trainable, "total_params": total, "resumed_from": resume}
    (EXP_DIR / "final_metrics.json").write_text(json.dumps(metrics, indent=2))
    del trainer, model
    free()
    return metrics

# ----------------------------------------------------------------------------------------
# analysis + report
# ----------------------------------------------------------------------------------------
def acc_table(base, ft, key):
    groups = defaultdict(list)
    for b, f in zip(base, ft):
        groups[b[key]].append((b["correct"], f["correct"]))
    rows = []
    for g in sorted(groups):
        n = len(groups[g])
        ba = 100.0 * sum(x for x, _ in groups[g]) / n
        fa = 100.0 * sum(y for _, y in groups[g]) / n
        rows.append([g, n, "%.1f%%" % ba, "%.1f%%" % fa, "%+.1f pp" % (fa - ba)])
    return rows

def build_report(ctx):
    base, ft, bench = ctx["base"], ctx["ft"], ctx["bench"]
    nb, nf = sum(r["correct"] for r in base), sum(r["correct"] for r in ft)
    delta = nf - nb
    trans = Counter()
    for b, f in zip(base, ft):
        trans[("Correct" if b["correct"] else "Wrong") + " -> " + ("Correct" if f["correct"] else "Wrong")] += 1
    w2c, c2w = trans["Wrong -> Correct"], trans["Correct -> Wrong"]
    p = mcnemar_exact(c2w, w2c)
    d, lo, hi = paired_bootstrap_delta([r["correct"] for r in base], [r["correct"] for r in ft])
    m = ctx["metrics"]
    L = []
    L.append("# AgriBot Qwen3 5,000-example baseline report\n")
    L.append("Generated: %s\n" % datetime.now(timezone.utc).isoformat())
    L.append("## 1. Experiment objective\nDoes LoRA fine-tuning of `%s` on the 5,000 agriculture examples improve exact MCQ accuracy on the frozen 100-question benchmark versus the untouched base model? Baseline only: no DoRA, DPO, MoA, RAG, or extra data.\n" % MODEL_NAME)
    L.append("## 2. Model\n`%s` (verified against model ids found in the repo: %s). Base weights unchanged; the fine-tuned model is the same base plus the LoRA adapter at `%s`.\n" % (MODEL_NAME, ctx["repo"]["model_ids_in_repo"], FINAL_DIR))
    L.append("## 3. Dataset\n`%s` - used unmodified. SHA-256: `%s`.\n" % (DATASET_NAME, ctx["dataset_sha"]))
    st = ctx["stats"]
    L.append("## 4. Dataset statistics\n" + md_table(["Statistic", "Value"], [[k, v] for k, v in st.items()]) + "\n")
    L.append("## 5. Data split\nDeterministic 90/10 split, seed %d. Train: %d | Validation: %d | Benchmark: %d (separate, never used in training). Split ids saved to `split_ids.json`.\n" % (ctx["seed"], ctx["n_train"], ctx["n_val"], len(bench)))
    L.append("## 6. LoRA configuration\n" + md_table(["Setting", "Value", "Source"], [[k, HP[k][0], HP[k][1]] for k in ("lora_r", "lora_alpha", "lora_dropout", "target_modules", "use_dora")]) + "\nTrainable parameters: %s of %s.\n" % (m["trainable_params"], m["total_params"]))
    L.append("## 7. Quantization configuration\n4-bit NF4, double quantization, compute dtype %s, frozen quantized base, only LoRA parameters trained.\n" % ctx["hw"]["compute_dtype"])
    L.append("## 8. Training hyperparameters\n" + md_table(["Setting", "Value", "Source"], [[k, v[0], v[1]] for k, v in HP.items()]) + "\nSeed: %d.\n" % ctx["seed"])
    L.append("## 9. Hardware\n" + md_table(["Item", "Value"], [[k, v] for k, v in ctx["hw"].items()]) + "\n")
    L.append("## 10. Training metrics\n" + md_table(["Metric", "Value"], [[k, v] for k, v in m.items()]) + "\nLower loss does not by itself mean better benchmark accuracy; the benchmark below is the primary measurement.\n")
    L.append("## 11. Benchmark methodology\nExact MCQ accuracy on `%s` (SHA-256 `%s`). Same prompt, same options, same deterministic decoding (%s) for both models. The reply is mapped to an option index; unparseable replies are counted as incorrect and reported separately. The model never sees the gold answer.\n" % (BENCHMARK_NAME, ctx["bench_sha"], GEN))
    L.append("## 12. Base model results\n%d correct / %d incorrect = %.1f%% (unparseable: %d).\n" % (nb, 100 - nb, nb, sum(1 for r in base if not r["parsed"])))
    L.append("## 13. Fine-tuned results\n%d correct / %d incorrect = %.1f%% (unparseable: %d).\n" % (nf, 100 - nf, nf, sum(1 for r in ft if not r["parsed"])))
    L.append("## 14. Accuracy improvement\n" + md_table(["Model", "Correct", "Incorrect", "Accuracy"], [["Qwen3 Base", nb, 100 - nb, "%d%%" % nb], ["Qwen3 + LoRA 5000", nf, 100 - nf, "%d%%" % nf]]) + "\n\nAbsolute change: **%+d percentage points**. Paired bootstrap 95%% CI for the change: [%+.1f, %+.1f] pp. Exact McNemar p = %.4f (%d questions flipped to correct, %d flipped to wrong). n = 100, so small differences are within noise.\n" % (delta, 100 * lo, 100 * hi, p, w2c, c2w))
    L.append("## 15. Category results\n" + md_table(["Category", "N", "Base", "Fine-tuned", "Change"], acc_table(base, ft, "category")) + "\n")
    L.append("## 16. Difficulty results\n" + md_table(["Level", "N", "Base", "Fine-tuned", "Change"], acc_table(base, ft, "level")) + "\n")
    L.append("## 17. Question-level changes\n" + md_table(["Transition", "Count"], [[k, trans[k]] for k in ("Correct -> Correct", "Wrong -> Correct", "Correct -> Wrong", "Wrong -> Wrong")]) + "\nFull per-question table: `question_level_results.csv` / `.json`.\n")
    lk = ctx["leak"]
    L.append("## 18. Leakage audit\n" + md_table(["Check", "Result"], [[k, v] for k, v in lk.items() if k not in ("flagged", "exact_overlap_ids", "normalized_overlap_ids")]) + "\n")
    dist_b = Counter(r["pred_label"] for r in base); dist_f = Counter(r["pred_label"] for r in ft)
    gold_d = Counter(r["gold_label"] for r in base)
    L.append("## 19. Error analysis\n- Unparseable replies: base %d, fine-tuned %d.\n- Answer-position distribution (gold / base / fine-tuned): %s / %s / %s. A fine-tuned model that collapses onto one label would show up here.\n- Questions that became worse (Correct -> Wrong): %s.\n" % (sum(1 for r in base if not r["parsed"]), sum(1 for r in ft if not r["parsed"]), dict(gold_d), dict(dist_b), dict(dist_f), [b["question_id"] for b, f in zip(base, ft) if b["correct"] and not f["correct"]]))
    L.append("LLM judge: not run. It is a secondary analysis in the spec and cannot replace MCQ accuracy.\n")
    verdict = "improved" if delta > 0 else ("did not change" if delta == 0 else "decreased")
    L.append("## 20. Final conclusion\nOn this 100-question MCQ benchmark, fine-tuning %s accuracy: %d%% -> %d%% (%+d pp; McNemar p = %.4f). %s The result describes this benchmark only; it is not evidence of general agricultural ability.\n" % (verdict, nb, nf, delta, p, "The change is statistically distinguishable from zero." if p < 0.05 else "The change is not statistically distinguishable from zero at n = 100."))
    return "\n".join(L), nb, nf, delta

# ----------------------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="inspect, validate, leakage-check, split; no GPU work")
    args = ap.parse_args()
    t_start = time.time()

    print("== 1. Inspecting repository: %s" % REPO_ROOT)
    repo = inspect_repo(REPO_ROOT)
    print(json.dumps({k: v for k, v in repo.items() if k != "json_files"}, indent=2))
    ds_path, bm_path = REPO_ROOT / DATASET_NAME, REPO_ROOT / BENCHMARK_NAME
    for pth in (ds_path, bm_path):
        if not pth.exists():
            die("Required file not found: %s" % pth)
    if repo["model_ids_in_repo"] and MODEL_NAME not in repo["model_ids_in_repo"]:
        die("MODEL_NAME %s does not appear in any repo script (found: %s). Not substituting a model."
            % (MODEL_NAME, repo["model_ids_in_repo"]))
    if FINAL_DIR.joinpath("adapter_config.json").exists():
        print("NOTE: a finished adapter already exists at %s - it will NOT be overwritten or retrained; "
              "evaluation will reuse it." % FINAL_DIR)
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    (EXP_DIR / "repo_inspection.json").write_text(json.dumps(repo, indent=2))

    rec_seed = [int(v) for _, v in repo["recorded_hyperparameters"].get("seed", [])]
    seed = rec_seed[0] if rec_seed and len(set(rec_seed)) == 1 else SEED_DEFAULT
    set_seeds(seed)
    print("Seed: %d (%s)" % (seed, "recorded in repo" if rec_seed and len(set(rec_seed)) == 1 else "default"))

    print("\n== 2. Dataset")
    rows = load_dataset_rows(ds_path)
    tokenizer = None
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print("Tokenizer unavailable (%s) - skipping token-length statistics." % e)
    stats, malformed, empty = dataset_statistics(rows, HP["max_seq_length"][0], tokenizer)
    print(json.dumps(stats, indent=2))
    (EXP_DIR / "dataset_statistics.json").write_text(json.dumps(stats, indent=2))
    if malformed or empty:
        die("Dataset has %d malformed and %d empty/missing-field records. The spec forbids removing or "
            "rewriting examples, so nothing is trained until you decide how to handle them."
            % (len(malformed), len(empty)))
    if stats["n_examples"] != 5000:
        print("WARNING: expected 5,000 examples, found %d." % stats["n_examples"])

    print("\n== 3. Benchmark")
    bench = load_benchmark(bm_path)
    bench_sha = sha256_of(bm_path)
    hash_file = bm_path.with_name("crop_100_hash.txt")
    if hash_file.exists() and hash_file.read_text().strip() != bench_sha:
        die("Benchmark hash differs from crop_100_hash.txt - the frozen benchmark has been modified.")
    print("100 MCQ questions OK. SHA-256: %s" % bench_sha)

    print("\n== 4. Leakage check")
    leak = leakage_check(bench, rows)
    (EXP_DIR / "leakage_report.json").write_text(json.dumps(leak, indent=2))
    print(json.dumps({k: v for k, v in leak.items() if k != "flagged"}, indent=2))
    stop_near = leak["near_duplicate_count_ge_%.2f" % NEAR_DUP_STOP_COS]
    if leak["exact_overlap_count"] or leak["normalized_overlap_count"] or stop_near:
        die("Benchmark leakage detected (exact %d, normalized %d, near-duplicate >= %.2f: %d). Nothing was "
            "removed or altered. Review %s and decide how to proceed." % (
                leak["exact_overlap_count"], leak["normalized_overlap_count"], NEAR_DUP_STOP_COS,
                stop_near, EXP_DIR / "leakage_report.json"))

    print("\n== 5. Deterministic 90/10 split")
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    n_val = int(round(HP["val_fraction"][0] * len(rows)))
    val_idx, train_idx = sorted(idx[:n_val]), sorted(idx[n_val:])
    train_rows, val_rows = [rows[i] for i in train_idx], [rows[i] for i in val_idx]
    (EXP_DIR / "split_ids.json").write_text(json.dumps({"seed": seed, "train_indices": train_idx,
                                                       "val_indices": val_idx}))
    print("Train %d | Validation %d | Benchmark %d" % (len(train_rows), len(val_rows), len(bench)))

    if args.dry_run:
        print("\nDry run complete: inspection, dataset, benchmark, leakage and split all passed. "
              "No model was loaded and nothing was trained.")
        return

    import torch
    if not torch.cuda.is_available():
        die("No CUDA GPU detected (check nvidia-smi and your CUDA-enabled PyTorch build).")
    import transformers, peft, trl
    _, dt = bnb_config()
    hw = {"gpu": torch.cuda.get_device_name(0),
          "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1),
          "cuda": torch.version.cuda, "torch": torch.__version__, "transformers": transformers.__version__,
          "peft": peft.__version__, "trl": trl.__version__, "compute_dtype": str(dt).replace("torch.", "")}

    print("\n== 6. BASE model benchmark (before any training)")
    base = evaluate("base", bench, tokenizer, adapter_dir=None)
    print("Base: %d/100" % sum(r["correct"] for r in base))

    print("\n== 7. Configuration summary")
    summary = {"model": MODEL_NAME, "dataset_examples": len(rows), "train": len(train_rows),
               "validation": len(val_rows), "benchmark": len(bench), "seed": seed,
               "precision": hw["compute_dtype"],
               "quantization": "4-bit NF4, double quant, compute dtype " + hw["compute_dtype"],
               "hyperparameters": {k: v[0] for k, v in HP.items()},
               "hyperparameter_sources": {k: v[1] for k, v in HP.items()},
               "effective_batch_size": HP["per_device_train_batch_size"][0] * HP["gradient_accumulation_steps"][0],
               "generation": GEN}
    print(json.dumps(summary, indent=2))
    (EXP_DIR / "training_config.json").write_text(json.dumps(summary, indent=2))

    if FINAL_DIR.joinpath("adapter_config.json").exists():
        print("\n== 8. Training skipped: adapter already exists (not overwritten).")
        mp = EXP_DIR / "final_metrics.json"
        metrics = json.loads(mp.read_text()) if mp.exists() else {"note": "metrics file not found"}
        for k in ("trainable_params", "total_params"):
            metrics.setdefault(k, "n/a")
    else:
        print("\n== 8. Training")
        metrics = train(train_rows, val_rows, tokenizer, seed, hw)
        print(json.dumps(metrics, indent=2))

    print("\n== 9. FINE-TUNED model benchmark (adapter reloaded from disk)")
    ft = evaluate("finetuned", bench, tokenizer, adapter_dir=FINAL_DIR)

    print("\n== 10. Report")
    ctx = {"base": base, "ft": ft, "bench": bench, "metrics": metrics, "leak": leak, "stats": stats,
           "repo": repo, "hw": hw, "seed": seed, "n_train": len(train_rows), "n_val": len(val_rows),
           "dataset_sha": sha256_of(ds_path), "bench_sha": bench_sha}
    text, nb, nf, delta = build_report(ctx)
    REPORT_PATH.write_text(text, encoding="utf-8")

    qrows = []
    qmap = {b["question_id"]: b for b in bench}
    for b, f in zip(base, ft):
        q = qmap[b["question_id"]]
        qrows.append({"question_id": b["question_id"], "question": q["question"], "gold_answer": b["gold_label"],
                      "base_answer": b["pred_label"], "base_correct": b["correct"],
                      "finetuned_answer": f["pred_label"], "finetuned_correct": f["correct"],
                      "category": b["category"], "level": b["level"],
                      "change": ("Correct" if b["correct"] else "Wrong") + " -> " + ("Correct" if f["correct"] else "Wrong")})
    (EVAL_DIR / "question_level_results.json").write_text(json.dumps(qrows, indent=2))
    import csv
    with open(EVAL_DIR / "question_level_results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(qrows[0].keys()))
        w.writeheader(); w.writerows(qrows)
    (EVAL_DIR / "benchmark_metrics.json").write_text(json.dumps(
        {"base_correct": nb, "finetuned_correct": nf, "delta_pp": delta, "generation": GEN}, indent=2))

    fmt = lambda v: "n/a" if v is None else (("%.4f" % v) if isinstance(v, float) else str(v))
    print("\n========================================\nAGRIBOT QWEN3 5000 BASELINE\n========================================")
    print("Base Model:\n%s\n\nTraining Examples:\n%d\n\nBenchmark Questions:\n%d\n" % (MODEL_NAME, len(rows), len(bench)))
    print("Base Accuracy:\n%d/100 = %d%%\n\nFine-Tuned Accuracy:\n%d/100 = %d%%\n" % (nb, nb, nf, nf))
    print("Improvement:\n%+d percentage points\n" % delta)
    print("Final Train Loss:\n%s\n\nFinal Validation Loss:\n%s\n" % (fmt(metrics.get("final_train_loss")), fmt(metrics.get("final_val_loss"))))
    print("Benchmark Leakage:\nexact=%d normalized=%d near-duplicate(>=%.2f)=%d\n========================================"
          % (leak["exact_overlap_count"], leak["normalized_overlap_count"], NEAR_DUP_REPORT_COS,
             leak["near_duplicate_count_ge_%.2f" % NEAR_DUP_REPORT_COS]))
    print("\n1. final adapter:          %s" % FINAL_DIR)
    print("2. training configuration: %s" % (EXP_DIR / "training_config.json"))
    print("3. training logs:          %s" % (EXP_DIR / "training_log.json"))
    print("4. benchmark predictions:  %s, %s" % (EVAL_DIR / "base_predictions.json", EVAL_DIR / "finetuned_predictions.json"))
    print("5. benchmark metrics:      %s" % (EVAL_DIR / "benchmark_metrics.json"))
    print("6. leakage report:         %s" % (EXP_DIR / "leakage_report.json"))
    print("7. final report:           %s" % REPORT_PATH)
    print("\nTotal wall-clock: %.1f min" % ((time.time() - t_start) / 60))

if __name__ == "__main__":
    main()
