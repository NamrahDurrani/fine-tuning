# %% [markdown]
# # AgriBot NB1 — Qwen3-4B-Thinking-2507 QLoRA / DoRA (local GPU, VS Code)
#
# Converted from the Colab notebook to run on your own machine. Open this file in VS Code
# with the Python extension installed and use **"Run Cell" / "Run Below"** above each `# %%`
# marker — VS Code treats those the same way Jupyter treats notebook cells, so you get the same
# step-by-step interactive experience as the original notebook, just as a plain `.py` file.
#
# **Before running cell 1:** create a virtual environment and install dependencies from the
# accompanying `requirements.txt` (not a `!pip install` cell — that was Colab-only magic):
# ```bash
# python -m venv .venv
# .venv\Scripts\activate          # Windows
# source .venv/bin/activate       # macOS/Linux
# pip install -r requirements.txt
# ```
# Install PyTorch **first, separately**, using the command for your CUDA version from
# https://pytorch.org/get-started/locally/ — `requirements.txt` intentionally does not pin
# `torch`, since the right build depends on your GPU driver/CUDA version.
#
# **Windows note:** `bitsandbytes` (used for 4-bit quantization) has had a rockier history on
# native Windows than on Linux/WSL2. Recent versions (0.43+) added native Windows wheels, but if
# `pip install bitsandbytes` or the model-loading cell below fails with a CUDA-setup error on
# Windows, the most reliable fix is running this under **WSL2** (Ubuntu) instead of native
# Windows Python — same GPU, Linux-native bitsandbytes build.
#
# To use the non-thinking sibling model, set `MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"` in the
# config cell below.

# %% [markdown]
# ## 1. Dependencies (install via requirements.txt in a terminal — see note above, not a cell)

# %% [markdown]
# ## 2. Local project directories (replaces Colab's Drive mount)

# %%
import os, json, glob, inspect, random
from pathlib import Path

# Edit this to wherever you want checkpoints/outputs stored on your machine.
# Defaults to a folder next to this script so it works out of the box.
PROJECT_DIR = Path(__file__).resolve().parent / "AgriBot_Project"
OUTPUT_DIR  = PROJECT_DIR / "weights_nb1"           # checkpoints land here
FINAL_DIR   = OUTPUT_DIR / "final_adapter"          # final adapter
PLOT_PATH   = OUTPUT_DIR / "loss_curves.png"
for d in (PROJECT_DIR, OUTPUT_DIR, FINAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# Optional: point the Hugging Face cache at a specific drive if your C: (or home) drive is
# low on space -- the base model download is several GB. Uncomment and edit if needed:
# os.environ["HF_HOME"] = str(Path("D:/hf_cache"))  # Windows example
# os.environ["HF_HOME"] = "/mnt/data/hf_cache"       # Linux example

print("PROJECT_DIR:", PROJECT_DIR)
print("OUTPUT_DIR: ", OUTPUT_DIR)
print("FINAL_DIR:  ", FINAL_DIR)

# %% [markdown]
# ## 3. Imports + config

# %%
import numpy as np
import torch
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

if not torch.cuda.is_available():
    raise SystemExit(
        "STOP: no CUDA GPU detected by PyTorch.\n"
        "Checklist:\n"
        "  1. Run `nvidia-smi` in a terminal -- if that fails, install/update your NVIDIA driver.\n"
        "  2. Confirm you installed a CUDA-enabled PyTorch build, not the CPU-only wheel:\n"
        "     https://pytorch.org/get-started/locally/ (pick your CUDA version there).\n"
        "  3. On Windows, consider WSL2 if you hit further native-Windows GPU library issues."
    )
print("GPU:", torch.cuda.get_device_name(0))
print(f"CUDA capability: {torch.cuda.get_device_capability(0)}")

MODEL_NAME   = "Qwen/Qwen3-4B-Thinking-2507"
SEED         = 42
MAX_LEN      = 1024
USE_DORA     = True     # True = DoRA (slightly better, ~15-25% slower). False = plain LoRA.
USE_BF16     = torch.cuda.is_bf16_supported()   # most RTX 30xx/40xx & datacenter cards -> bf16; older cards -> fp16
COMPUTE_DTYPE = torch.bfloat16 if USE_BF16 else torch.float16

# Where your dataset lives locally. Put the JSON file at this path (or edit the path).
DATA_PATH    = PROJECT_DIR / "AgriBot_5000_Unique_Agriculture_Dataset.json"
ALLOW_SIMULATED_DATA = False   # set True only to smoke-test the pipeline

SYSTEM_PROMPT = ("You are AgriBot, an agricultural assistant. Give accurate, practical, "
                 "safe advice and defer to local extension guidance for product-specific decisions.")

random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# %% [markdown]
# ## 4. Load dataset (real JSON, or simulated fallback) and split 90/10

# %%
def load_real_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):  # tolerate {"data": [...]} / {"examples": [...]}
        raw = next(v for v in raw.values() if isinstance(v, list))
    rows = [r for r in raw if r.get("instruction") and r.get("response")]
    return rows

def simulate_dataset(n=5000):
    domains = {
        "Crop production": ["wheat", "rice", "maize", "cotton", "sugarcane"],
        "Pests & diseases": ["aphids", "rice blast", "whitefly", "stem borer", "leaf rust"],
        "Soil & fertilizers": ["nitrogen", "phosphorus", "soil pH", "organic matter", "salinity"],
        "Pakistan agriculture": ["Punjab wheat", "Sindh cotton", "Kharif season", "Rabi season", "canal irrigation"],
    }
    rows, i = [], 0
    while len(rows) < n:
        cat = list(domains)[i % 4]
        topic = domains[cat][(i // 4) % 5]
        rows.append({"id": f"SIM_{i:05d}", "category": cat,
                     "instruction": f"[{i}] What should a farmer know about {topic}?",
                     "response": f"Simulated guidance #{i} about {topic} in the area of {cat}."})
        i += 1
    return rows

if DATA_PATH.exists():
    rows = load_real_dataset(DATA_PATH)
    print(f"Loaded {len(rows)} usable rows from {DATA_PATH}")
    # Guard against a truncated / wrong file
    assert len(rows) >= 4000, f"Only {len(rows)} rows found - this does not look like the full dataset."
elif ALLOW_SIMULATED_DATA:
    rows = simulate_dataset()
    print("WARNING: using SIMULATED data")
else:
    raise FileNotFoundError(f"Put the dataset at {DATA_PATH} (or set ALLOW_SIMULATED_DATA=True).")

keep = ["instruction", "response", "category"]
ds = Dataset.from_list([{k: r.get(k, "") for k in keep} for r in rows])
split = ds.train_test_split(test_size=0.10, seed=SEED, shuffle=True)
train_raw, val_raw = split["train"], split["test"]
print(f"Train: {len(train_raw)} | Validation: {len(val_raw)}")
if "category" in ds.column_names:
    from collections import Counter
    print("Category mix (train):", dict(Counter(train_raw["category"])))

# %% [markdown]
# ## 5. Tokenizer + prompt formatting

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Qwen3-Thinking-2507 opens every assistant turn with "<think>\n". Our dataset has
# no reasoning traces, so we train it to emit an empty think block, then the answer.
def to_prompt_completion(ex):
    prompt = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT},
         {"role": "user", "content": ex["instruction"].strip()}],
        tokenize=False, add_generation_prompt=True)
    if not prompt.rstrip().endswith("<think>"):
        prompt += "<think>\n"
    completion = "\n</think>\n\n" + ex["response"].strip() + "<|im_end|>\n"
    return {"prompt": prompt, "completion": completion}

train_ds = train_raw.map(to_prompt_completion, remove_columns=train_raw.column_names)
val_ds   = val_raw.map(to_prompt_completion,   remove_columns=val_raw.column_names)
print("\n--- Sample training text ---\n", train_ds[0]["prompt"] + train_ds[0]["completion"])

# %% [markdown]
# ## 6. 4-bit quantization (NF4 + double quant)

# %%
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=COMPUTE_DTYPE,
)

try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=COMPUTE_DTYPE,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
except OSError as e:
    raise SystemExit(
        f"Could not download/load {MODEL_NAME} ({e}).\n"
        "If this is a 401/403 auth error, run `huggingface-cli login` in your terminal first "
        "(or set the HF_TOKEN environment variable) and re-run this cell.\n"
        "This also downloads several GB on first run -- check you have disk space / a stable "
        "connection, or set HF_HOME above to a drive with more room."
    )
model.config.use_cache = False
model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# %% [markdown]
# ## 7. LoRA / DoRA config

# %%
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    use_dora=USE_DORA,
)

# %% [markdown]
# ## 8. Training arguments + SFTTrainer

# %%
# Effective batch = 2 x 8 = 16  ->  ~253 steps/epoch on 4,500 examples
sft_kwargs = dict(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    weight_decay=0.01,
    max_grad_norm=0.3,
    optim="paged_adamw_8bit",
    fp16=not USE_BF16,
    bf16=USE_BF16,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    logging_steps=10,
    eval_steps=100,
    save_steps=500,                 # checkpoint every 500 steps
    save_total_limit=3,
    report_to="none",
    seed=SEED,
    completion_only_loss=True,      # loss only on the answer (prompt/completion format)
    dataset_num_proc=2,             # belongs in SFTConfig, not SFTTrainer
)

# transformers/trl rename or remove arguments across versions - handle both
cfg_params = inspect.signature(SFTConfig.__init__).parameters

# warmup: old versions use warmup_ratio; newer ones take a float < 1 in warmup_steps
if "warmup_ratio" in cfg_params:
    sft_kwargs["warmup_ratio"] = 0.03
else:
    sft_kwargs["warmup_steps"] = 0.03   # float < 1 is treated as a ratio of total steps

sft_kwargs["eval_strategy" if "eval_strategy" in cfg_params else "evaluation_strategy"] = "steps"
sft_kwargs["max_length" if "max_length" in cfg_params else "max_seq_length"] = MAX_LEN

# drop anything this installed version doesn't recognise (and say so)
dropped = [k for k in sft_kwargs if k not in cfg_params]
if dropped:
    print("Skipping unsupported SFTConfig args:", dropped)
sft_kwargs = {k: v for k, v in sft_kwargs.items() if k in cfg_params}

training_args = SFTConfig(**sft_kwargs)

trainer_kwargs = dict(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=peft_config,
)

if "processing_class" in inspect.signature(SFTTrainer.__init__).parameters:
    trainer_kwargs["processing_class"] = tokenizer
else:
    trainer_kwargs["tokenizer"] = tokenizer

trainer = SFTTrainer(**trainer_kwargs)
trainer.model.print_trainable_parameters()

# %% [markdown]
# ## 9. Train (auto-resumes if interrupted and a checkpoint exists in OUTPUT_DIR)

# %%
ckpts = sorted(glob.glob(str(OUTPUT_DIR / "checkpoint-*")),
               key=lambda p: int(p.split("-")[-1]))
resume = ckpts[-1] if ckpts else None
print("Resuming from:", resume)
trainer.train(resume_from_checkpoint=resume)

# %% [markdown]
# ## 10. Save final adapter locally

# %%
trainer.model.save_pretrained(str(FINAL_DIR))        # adapter weights only (small)
tokenizer.save_pretrained(str(FINAL_DIR))
with open(FINAL_DIR / "run_config.json", "w") as f:
    json.dump({"base_model": MODEL_NAME, "dora": USE_DORA, "r": 16, "alpha": 32,
               "dropout": 0.05, "lr": 2e-4, "epochs": 3, "eff_batch": 16,
               "max_len": MAX_LEN, "seed": SEED,
               "train_examples": len(train_ds), "val_examples": len(val_ds)}, f, indent=2)
with open(OUTPUT_DIR / "log_history.json", "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)
print("Adapter saved to:", FINAL_DIR)

# %% [markdown]
# ## 11. Training / validation loss curves

# %%
logs = trainer.state.log_history
tr = [(l["step"], l["loss"]) for l in logs if "loss" in l]
ev = [(l["step"], l["eval_loss"]) for l in logs if "eval_loss" in l]

plt.figure(figsize=(9, 5))
plt.plot(*zip(*tr), label="Training loss", alpha=0.8)
if ev:
    plt.plot(*zip(*ev), marker="o", label="Validation loss")
plt.xlabel("Step"); plt.ylabel("Loss")
plt.title("AgriBot NB1 - Qwen3-4B-Thinking-2507 QLoRA" + (" + DoRA" if USE_DORA else ""))
plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
plt.savefig(PLOT_PATH, dpi=150)
plt.show()

print("\nStep | Train loss")
for s, l in tr[::max(1, len(tr)//10)]: print(f"{s:5d} | {l:.4f}")
print("\nStep | Val loss")
for s, l in ev: print(f"{s:5d} | {l:.4f}")
if ev: print(f"\nBest val loss: {min(l for _, l in ev):.4f}")

# %% [markdown]
# ## 12. Quick sanity check (generation with the fine-tuned adapter)

# %%
model.eval(); model.config.use_cache = True
test_q = "Why is crop rotation useful for wheat in Punjab?"
prompt = tokenizer.apply_chat_template(
    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": test_q}],
    tokenize=False, add_generation_prompt=True)
if not prompt.rstrip().endswith("<think>"):
    prompt += "<think>\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=300, do_sample=False,
                         pad_token_id=tokenizer.pad_token_id)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))

# %% [markdown]
# ## 13. (Optional) Before/after comparison on your CROP questions
#
# Put your real 100 CROP questions locally as a JSON list (strings, or dicts with a `question`
# key) at the path below. This generates answers from the **base** model (adapter disabled) and
# the **fine-tuned** model and saves both locally for blinded judging. Nothing is invented here:
# if the file is missing, the cell just skips.

# %%
CROP_PATH = PROJECT_DIR / "crop_100_questions.json"
EVAL_OUT  = OUTPUT_DIR / "crop_base_vs_finetuned.json"

def gen(question, max_new_tokens=400):
    p = tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True)
    if not p.rstrip().endswith("<think>"):
        p += "<think>\n"
    x = tokenizer(p, return_tensors="pt").to("cuda")
    with torch.no_grad():
        o = model.generate(**x, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(o[0][x["input_ids"].shape[1]:], skip_special_tokens=True).strip()

if CROP_PATH.exists():
    qs = json.load(open(CROP_PATH, encoding="utf-8"))
    qs = [q["question"] if isinstance(q, dict) else q for q in qs]
    print(f"{len(qs)} CROP questions")
    results = []
    for i, q in enumerate(qs):
        with model.disable_adapter():          # base Qwen3, no LoRA
            base_ans = gen(q)
        ft_ans = gen(q)                         # fine-tuned
        results.append({"id": i, "question": q, "base": base_ans, "finetuned": ft_ans})
        if (i + 1) % 10 == 0: print(f"  {i+1}/{len(qs)}")
    json.dump(results, open(EVAL_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("Saved:", EVAL_OUT)
else:
    print("CROP file not found - skipping. Expected:", CROP_PATH)
