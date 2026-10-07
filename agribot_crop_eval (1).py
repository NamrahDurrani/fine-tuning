# %% [markdown]
# # AgriBot - CROP-100 evaluation (Base vs Fine-Tuned vs MoA), leakage audit, blinded LLM judge
#
# Runs on a local GPU PC (Windows/Linux) or Google Colab. Put this file in the SAME folder as
# your dataset JSON / agribot_nb1_local.py / weights_nb1 folder and run it:
#
#     python agribot_crop_eval.py          (or "Run Cell" in VS Code, cell by cell)
#
# Extra packages (install PyTorch first, per pytorch.org, then):
#     pip install transformers peft bitsandbytes accelerate datasets sentence-transformers
#     pip install google-genai pandas numpy scipy scikit-learn matplotlib tqdm requests
#   (see requirements_eval.txt and .env.example next to this file)
#
# Re-running is safe: every generation / judgement is checkpointed to
# evaluation_reports/evaluation_checkpoint.json and finished work is never repeated.

# %% [markdown]
# ## 1. CONFIGURATION  (the only block you normally need to edit)

# %%
from __future__ import annotations

import gc
import hashlib
import itertools
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ============================ EDIT ONLY THIS BLOCK ============================
PROJECT_DIR: Optional[str] = None          # None = auto-detect (folder of this script, or ./AgriBot_Project)
NOTEBOOK_1_WEIGHTS: Optional[str] = None   # None = auto-discover (weights_nb1/final_adapter)
NOTEBOOK_2_WEIGHTS: Optional[str] = None   # None = auto-discover (weights_nb2/...). Not found -> Model C skipped
GEMINI_API_KEY = ""                        # judge 1 (or env GEMINI_API_KEY / GOOGLE_API_KEY, or .env file)
GPTOSS_PROVIDER = "groq"                   # judge 2 host: "groq" / "openrouter" / "custom"
GPTOSS_API_KEY = ""                        # judge 2 (or env GROQ_API_KEY / OPENROUTER_API_KEY / GPTOSS_API_KEY, or .env)
# ==============================================================================

ENVIRONMENT = "auto"                       # auto / colab / local
# --- judges: tried in JUDGE_ORDER; when one hits its daily quota (or its key/model is rejected) the next one takes over
JUDGE_ORDER = ["gemini", "gptoss"]
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_MIN_INTERVAL_SECONDS = 7.0          # spacing between calls (~8/min, under a 10 RPM cap)
GPTOSS_MODEL = None                        # None = host default (groq: openai/gpt-oss-20b)
GPTOSS_BASE_URL = None                     # None = host default
GPTOSS_MIN_INTERVAL_SECONDS = 8.0          # Groq free tier has a small tokens-per-minute cap; 429s are retried anyway
GPTOSS_PRESETS = {
    "groq": {"base_url": "https://api.groq.com/openai/v1", "model": "openai/gpt-oss-20b",
             "env": ["GROQ_API_KEY", "GPTOSS_API_KEY"], "extras": {"reasoning_effort": "medium"}},
    "openrouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-oss-20b:free",
                   "env": ["OPENROUTER_API_KEY", "GPTOSS_API_KEY"], "extras": {}},
    "custom": {"base_url": None, "model": None, "env": ["GPTOSS_API_KEY"], "extras": {}},
}

BASE_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
DATASET_FILENAME = "AgriBot_5000_Unique_Agriculture_Dataset.json"
EXTRA_MODELS: dict = {}                    # e.g. {"new_sft": "weights_nb3/final_adapter", "sft_dpo": "..."}
QUANTIZATION = "4bit"                      # "4bit" (NF4) or "none"
ALLOW_CPU = False                          # never silently falls back to CPU unless you set this True

MAX_NEW_TOKENS = 1024                      # identical for every model (do_sample=False == temperature 0)
TARGET = 95
PREVIOUS_BEST = 88                         # your earlier result, only used for the "gap" display
EXPECTED_QUESTIONS = 100

RUN_GENERATION = True
RUN_JUDGE = True
RUN_DATASET_TOOLS = True                   # dataset quality report + grouped split suggestion
RAG_DOCS_DIR: Optional[str] = None         # folder of .txt/.md/.json/.jsonl docs -> also run "rag" mode
RAG_TOP_K = 3

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
NEAR_THRESHOLDS = (0.90, 0.95, 0.97)
RUN_PAIRWISE = True                        # set False to save ~1/3 of judge calls (free-tier friendly)
RUN_SECOND_JUDGE = True                    # set False to judge each answer once only
SECOND_JUDGE_BAND = (45, 65)               # 2nd judge if first total is in this band ...
SECOND_JUDGE_MIN_CONF = 0.70               # ... or if correct=True but confidence below this
DISAGREE_SCORE_DELTA = 20                  # judges differing by more than this -> disagreement
SEED = 42

# Must be IDENTICAL to the prompt used in agribot_nb1_local.py training.
SYSTEM_PROMPT = ("You are AgriBot, an agricultural assistant. Give accurate, practical, "
                 "safe advice and defer to local extension guidance for product-specific decisions.")

ERROR_CATEGORIES = ["FACTUAL_ERROR", "INCOMPLETE", "WRONG_CROP", "WRONG_REGION", "WRONG_SEASON",
                    "WRONG_DISEASE", "WRONG_SOIL_CHEMISTRY", "UNSUPPORTED_RECOMMENDATION",
                    "HALLUCINATION", "QUESTION_MISUNDERSTANDING"]
NO_ANSWER_CATEGORY = "NO_ANSWER_TRUNCATED"   # extra category: generation ended inside the thinking block

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    def tqdm(x, **kwargs):
        return x

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("agribot_eval")

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:  # interactive
    SCRIPT_DIR = Path.cwd()

# %% [markdown]
# ## 2. Small utilities

# %%
def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def atomic_write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=_json_default)
    os.replace(tmp, path)


def norm_text(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\u0600-\u06ff\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def clamp(x, lo, hi):
    try:
        x = float(x)
    except (TypeError, ValueError):
        x = 0.0
    return max(lo, min(hi, x))


def free_cuda():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


# %% [markdown]
# ## 3. Environment, paths and hardware report

# %%
def detect_environment() -> str:
    env = ENVIRONMENT.lower()
    if env in ("colab", "local"):
        return env
    import importlib.util
    try:
        return "colab" if importlib.util.find_spec("google.colab") is not None else "local"
    except (ImportError, ValueError):
        return "local"


class Paths:
    """Resolves every folder/file we need. Works on Colab (Drive) and local disks."""

    def __init__(self):
        self.env = detect_environment()
        if self.env == "colab" and not Path("/content/drive/MyDrive").exists():
            try:
                from google.colab import drive  # type: ignore
                drive.mount("/content/drive")
            except Exception as e:
                log.warning("Could not mount Google Drive: %s", e)
        self.project = self._project_dir()
        cands = [self.project, SCRIPT_DIR, SCRIPT_DIR / "AgriBot_Project", Path.cwd()]
        self.roots: list[Path] = []
        for c in cands:
            if c.exists() and c.resolve() not in [r.resolve() for r in self.roots]:
                self.roots.append(c)
        self.eval_dir = self.project / "evaluation_reports"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(self.eval_dir / "evaluation.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        log.addHandler(fh)
        self.dataset = self._find_dataset()

    def _project_dir(self) -> Path:
        if PROJECT_DIR:
            return Path(PROJECT_DIR).expanduser()
        if self.env == "colab":
            return Path("/content/drive/MyDrive/AgriBot_Project")
        for c in (SCRIPT_DIR, SCRIPT_DIR / "AgriBot_Project", Path.cwd(), Path.cwd() / "AgriBot_Project"):
            if (c / DATASET_FILENAME).exists() or any(c.glob("weights_nb*")):
                return c
        return SCRIPT_DIR

    def _find_dataset(self) -> Optional[Path]:
        for r in self.roots:
            if (r / DATASET_FILENAME).exists():
                return r / DATASET_FILENAME
        return None

    def resolve_user_path(self, p: str) -> Path:
        path = Path(p).expanduser()
        if path.is_absolute():
            return path
        for r in self.roots:
            if (r / path).exists():
                return r / path
        return self.roots[0] / path


def hardware_report() -> dict:
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            free_b, total_b = torch.cuda.mem_get_info(0)
            info.update(gpu=props.name, vram_total_gb=round(total_b / 2**30, 2),
                        free_vram_gb=round(free_b / 2**30, 2),
                        bf16_supported=bool(torch.cuda.is_bf16_supported()),
                        compute_capability=torch.cuda.get_device_capability(0))
            info["dtype"] = "bfloat16" if info["bf16_supported"] else "float16"
        else:
            info.update(gpu=None, vram_total_gb=0.0, free_vram_gb=0.0, dtype="float32")
    except ImportError:
        info.update(torch=None, cuda_available=False, gpu=None, vram_total_gb=0.0, free_vram_gb=0.0, dtype=None)
    for mod in ("transformers", "peft", "bitsandbytes", "sentence_transformers"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "installed")
        except Exception as e:
            info[mod] = f"NOT AVAILABLE ({type(e).__name__})"
    info["quantization_available"] = not str(info.get("bitsandbytes", "")).startswith("NOT")
    return info


def print_hardware(info: dict, paths: Paths) -> None:
    print("=" * 78)
    print("ENVIRONMENT REPORT")
    print("=" * 78)
    print(f"Environment:        {paths.env}")
    print(f"Project dir:        {paths.project}")
    print(f"Evaluation dir:     {paths.eval_dir}")
    print(f"Dataset:            {paths.dataset}")
    for k in ("python", "platform", "torch", "cuda_available", "cuda_version", "gpu", "vram_total_gb",
              "free_vram_gb", "dtype", "transformers", "peft", "bitsandbytes", "sentence_transformers",
              "quantization_available"):
        print(f"{k:<20}{info.get(k)}")
    print("=" * 78)


# %% [markdown]
# ## 4. Model discovery and validation (no silent assumptions)

# %%
SKIP_DIRS = {".venv", "venv", "env", "site-packages", "__pycache__", ".git", "node_modules", ".cache",
             "evaluation_reports"}
ADAPTER_WEIGHT_FILES = ("adapter_model.safetensors", "adapter_model.bin")


def _model_like(p: Path) -> bool:
    return p.is_dir() and ((p / "adapter_config.json").exists() or (p / "config.json").exists())


def _has_full_weights(p: Path) -> bool:
    return any(p.glob("*.safetensors")) or any(p.glob("pytorch_model*.bin"))


def pick_model_subdir(d: Path) -> Path:
    if _model_like(d):
        return d
    for name in ("final_adapter", "final", "final_model", "merged", "best"):
        if _model_like(d / name):
            return d / name
    ckpts = [c for c in d.glob("checkpoint-*") if _model_like(c)]
    if ckpts:
        return sorted(ckpts, key=lambda c: int(re.sub(r"\D", "", c.name) or 0))[-1]
    return d


def resolve_weights(configured: Optional[str], tag: str, paths: Paths) -> Optional[Path]:
    if configured:
        p = paths.resolve_user_path(configured)
        return pick_model_subdir(p) if p.exists() else p
    for r in paths.roots:
        d = r / f"weights_{tag}"
        if d.exists():
            return pick_model_subdir(d)
    return None


def inspect_model_dir(path: Path) -> dict:
    rep: dict[str, Any] = {"path": str(path), "kind": "unknown", "problems": [], "notes": []}
    if not path.exists() or not path.is_dir():
        rep["problems"].append("path does not exist or is not a directory")
        return rep
    try:
        if (path / "adapter_config.json").exists():
            cfg = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
            rep.update(kind="peft_adapter", base_model=cfg.get("base_model_name_or_path"),
                       peft_type=cfg.get("peft_type"), r=cfg.get("r"), lora_alpha=cfg.get("lora_alpha"),
                       use_dora=cfg.get("use_dora"), target_modules=cfg.get("target_modules"))
            if not any((path / f).exists() for f in ADAPTER_WEIGHT_FILES):
                rep["problems"].append("adapter_config.json found but no adapter_model.safetensors/.bin")
            rep["has_tokenizer_files"] = (path / "tokenizer.json").exists() or (path / "tokenizer_config.json").exists()
        elif (path / "config.json").exists():
            cfg = json.loads((path / "config.json").read_text(encoding="utf-8"))
            rep.update(kind="full_or_merged_hf_model", architectures=cfg.get("architectures"),
                       pre_quantized="quantization_config" in cfg)
            if not _has_full_weights(path):
                rep["problems"].append("config.json found but no *.safetensors / pytorch_model*.bin weights")
            rep["notes"].append("files alone cannot distinguish a merged model from an original full model")
            rep["weights_gb"] = round(sum(f.stat().st_size for f in itertools.chain(path.glob("*.safetensors"),
                                                                                     path.glob("pytorch_model*.bin"))) / 2**30, 2)
        elif any(path.glob("*.gguf")):
            rep.update(kind="gguf_unsupported")
            rep["problems"].append("GGUF is not supported by this evaluator (needs transformers/PEFT format)")
        else:
            kids = [str(c) for c in sorted(path.iterdir()) if _model_like(c)]
            rep["candidates"] = kids
            rep["problems"].append("folder is a container, not a model; point NOTEBOOK_*_WEIGHTS at one of: "
                                   + (", ".join(kids) if kids else "(no model-like subfolders found)"))
    except (json.JSONDecodeError, OSError) as e:
        rep["problems"].append(f"could not read config: {e}")
    return rep


def fingerprint(path: Path) -> str:
    parts = []
    for f in sorted(itertools.chain(path.glob("adapter_model.*"), path.glob("*.safetensors"),
                                    path.glob("pytorch_model*.bin"), path.glob("adapter_config.json"))):
        st = f.stat()
        parts.append(f"{f.name}:{st.st_size}:{int(st.st_mtime)}")
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class ModelSpec:
    def __init__(self, label: str, kind: str, path: Optional[Path], info: dict, role: str):
        self.label, self.kind, self.path, self.info, self.role = label, kind, path, info, role


def discover_models(paths: Paths) -> tuple[list[ModelSpec], list[dict]]:
    specs = [ModelSpec("base", "base", None, {"model": BASE_MODEL}, "Model A - Base")]
    skipped: list[dict] = []
    wanted = [("fine_tuned", NOTEBOOK_1_WEIGHTS, "nb1", "Model B - Fine-Tuned (Notebook 1)", True),
              ("moa", NOTEBOOK_2_WEIGHTS, "nb2", "Model C - Optimized MoA (Notebook 2)", False)]
    for label, cfg, tag, role, required in wanted:
        p = resolve_weights(cfg, tag, paths)
        if p is None:
            msg = f"{role}: no weights found (looked for weights_{tag} under {[str(r) for r in paths.roots]})"
            if required:
                raise SystemExit(msg + "\nSet NOTEBOOK_1_WEIGHTS at the top of the script.")
            skipped.append({"label": label, "reason": msg})
            continue
        rep = inspect_model_dir(p)
        if rep["problems"]:
            if required:
                raise SystemExit(f"{role}: {rep['problems']} (path: {p})")
            skipped.append({"label": label, "reason": f"{rep['problems']} (path: {p})"})
            continue
        specs.append(ModelSpec(label, rep["kind"], p, rep, role))
    for label, pth in EXTRA_MODELS.items():
        p = pick_model_subdir(paths.resolve_user_path(pth))
        rep = inspect_model_dir(p)
        if rep["problems"]:
            skipped.append({"label": label, "reason": f"{rep['problems']} (path: {p})"})
            continue
        specs.append(ModelSpec(label, rep["kind"], p, rep, f"Extra - {label}"))
    for s in specs:
        if s.kind == "peft_adapter":
            b = str(s.info.get("base_model") or "")
            if b and Path(b).name.lower() != Path(BASE_MODEL).name.lower():
                raise SystemExit(f"{s.label}: adapter was trained on '{b}' but BASE_MODEL is '{BASE_MODEL}'. "
                                 "Refusing to attach it to a different base model.")
    return specs, skipped


def print_model_report(specs: list[ModelSpec], skipped: list[dict]) -> None:
    print("=" * 78)
    print("MODEL LOADING REPORT")
    print("=" * 78)
    for s in specs:
        print(f"[{s.label}] {s.role}")
        print(f"   kind:  {s.kind}")
        print(f"   path:  {s.path if s.path else BASE_MODEL + ' (Hugging Face hub / local cache)'}")
        for k in ("base_model", "peft_type", "r", "lora_alpha", "use_dora", "target_modules", "pre_quantized",
                  "weights_gb", "has_tokenizer_files"):
            if k in s.info:
                print(f"   {k}: {s.info[k]}")
        for n in s.info.get("notes", []):
            print(f"   note: {n}")
    for sk in skipped:
        print(f"[SKIPPED] {sk['label']}: {sk['reason']}")
    print("=" * 78)


# %% [markdown]
# ## 5. Memory estimate, loading, generation

# %%
def estimate_params_b(name_or_path: str, weights_gb: Optional[float] = None, pre_quant: bool = False) -> float:
    m = re.search(r"(\d+(?:\.\d+)?)\s*[bB](?![a-zA-Z])", Path(name_or_path).name)
    if m:
        return float(m.group(1))
    if weights_gb:
        return weights_gb / (0.55 if pre_quant else 2.0)
    return 4.0


def estimate_memory_gb(params_b: float, quant: str) -> float:
    return params_b * 0.55 + 1.3 if quant == "4bit" else params_b * 2.0 + 1.3


def preflight_fit(label: str, params_b: float, quant: str, hw: dict) -> None:
    est = estimate_memory_gb(params_b, quant)
    free = hw.get("free_vram_gb") or 0
    if hw.get("cuda_available"):
        import torch
        free = torch.cuda.mem_get_info(0)[0] / 2**30
    ok = est <= free * 0.92
    print(f"GPU: {hw.get('gpu')}\nVRAM: {hw.get('vram_total_gb')} GB total / {free:.2f} GB free\n"
          f"Model: {label}\nEstimated memory: ~{est:.1f} GB\nQuantization: {quant}\n"
          f"Expected feasibility: {'OK' if ok else 'WILL NOT FIT'}\n")
    if not ok:
        raise RuntimeError(f"{label}: estimated {est:.1f} GB needed but only {free:.2f} GB VRAM is free. "
                           "Close other GPU programs (training notebooks hold VRAM!), or free memory and retry. "
                           "Not falling back to CPU (set ALLOW_CPU=True to allow that).")


class Runner:
    def __init__(self, tok, model, hw: dict, adapter_labels: list[str]):
        self.tok, self.model, self.hw = tok, model, hw
        self.cuda = bool(hw.get("cuda_available"))
        self.device = "cuda" if self.cuda else "cpu"
        self.adapters = adapter_labels
        eos = set()
        ge = getattr(model, "generation_config", None)
        raw = getattr(ge, "eos_token_id", None)
        for v in (raw if isinstance(raw, (list, tuple)) else [raw]):
            if v is not None:
                eos.add(int(v))
        if tok.eos_token_id is not None:
            eos.add(int(tok.eos_token_id))
        self.eos_ids = eos

    @contextmanager
    def use(self, label: str):
        if label == "base":
            if self.adapters:
                with self.model.disable_adapter():
                    yield
            else:
                yield
        else:
            if self.adapters:
                self.model.set_adapter(label)
            yield


def _dtype(hw: dict):
    import torch
    if not hw.get("cuda_available"):
        return torch.float32
    return torch.bfloat16 if hw.get("bf16_supported") else torch.float16


def load_tokenizer():
    from transformers import AutoTokenizer
    try:
        tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"Could not load tokenizer for {BASE_MODEL}: {e}. If this is an auth/network error "
                           "run `huggingface-cli login` or check your connection.") from e
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(name_or_path: str, quant: str, hw: dict, pre_quantized: bool = False):
    import torch
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    dtype = _dtype(hw)
    kw: dict[str, Any] = dict(torch_dtype=dtype, trust_remote_code=True, attn_implementation="sdpa")
    if hw.get("cuda_available"):
        kw["device_map"] = {"": 0}   # single-GPU "auto" that can never silently offload to CPU
        if quant == "4bit" and not pre_quantized:
            if not hw.get("quantization_available"):
                raise RuntimeError("bitsandbytes is not available; install it (on Windows it may need WSL2) "
                                   "or set QUANTIZATION='none' if your VRAM allows fp16/bf16.")
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=dtype)
    try:
        model = AutoModelForCausalLM.from_pretrained(name_or_path, **kw)
    except torch.cuda.OutOfMemoryError as e:
        free_cuda()
        raise RuntimeError(f"CUDA out of memory while loading {name_or_path}. Free GPU memory and retry.") from e
    except OSError as e:
        raise RuntimeError(f"Could not load {name_or_path}: {e}") from e
    model.eval()
    return model


def load_runner_for_group(group: list[ModelSpec], hw: dict) -> Runner:
    """Base model once + every PEFT adapter attached by name (base eval = adapter disabled)."""
    quant = QUANTIZATION if hw.get("cuda_available") else "none"
    preflight_fit("+".join(s.label for s in group), estimate_params_b(BASE_MODEL), quant, hw)
    tok = load_tokenizer()
    model = load_causal_lm(BASE_MODEL, quant, hw)
    adapters = [s for s in group if s.kind == "peft_adapter"]
    if adapters:
        from peft import PeftModel
        for i, s in enumerate(adapters):
            if i == 0:
                model = PeftModel.from_pretrained(model, str(s.path), adapter_name=s.label)
            else:
                model.load_adapter(str(s.path), adapter_name=s.label)
        model.eval()
    return Runner(tok, model, hw, [s.label for s in adapters])


def load_runner_for_full(spec: ModelSpec, hw: dict) -> Runner:
    quant = QUANTIZATION if hw.get("cuda_available") else "none"
    pre_q = bool(spec.info.get("pre_quantized"))
    params = estimate_params_b(spec.path.name, spec.info.get("weights_gb"), pre_q)
    preflight_fit(spec.label, params, "4bit" if pre_q else quant, hw)
    tok = load_tokenizer()
    model = load_causal_lm(str(spec.path), quant, hw, pre_quantized=pre_q)
    return Runner(tok, model, hw, [])


def build_prompt(tok, question: str, context: Optional[str] = None) -> str:
    user = question.strip()
    if context:
        user = ("Use the reference material below only if it is relevant to the question.\n\n"
                f"REFERENCE MATERIAL:\n{context}\n\nQUESTION:\n{question.strip()}")
    p = tok.apply_chat_template([{"role": "system", "content": SYSTEM_PROMPT},
                                 {"role": "user", "content": user}],
                                tokenize=False, add_generation_prompt=True)
    if not p.rstrip().endswith("<think>"):      # same convention as the training script
        p += "<think>\n"
    return p


def split_reasoning(raw: str, finish: str) -> tuple[str, str, bool]:
    """-> (visible reasoning text, final answer, think_block_closed)"""
    raw = raw.replace("<think>", "")
    if "</think>" in raw:
        think, ans = raw.split("</think>", 1)
        return think.strip(), ans.strip(), True
    if finish == "stop":
        return "", raw.strip(), True
    return raw.strip(), "", False               # ran out of tokens while still thinking


def generate_one(runner: Runner, prompt: str) -> dict:
    import torch
    tok, model = runner.tok, runner.model
    inputs = tok(prompt, return_tensors="pt").to(runner.device)
    n_prompt = int(inputs["input_ids"].shape[1])
    if runner.cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.inference_mode():
        out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, temperature=None,
                             top_p=None, top_k=None, pad_token_id=tok.pad_token_id, use_cache=True)
    if runner.cuda:
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ids = out[0][n_prompt:].tolist()
    finish = "stop" if (ids and ids[-1] in runner.eos_ids) else "length"
    raw = tok.decode(ids, skip_special_tokens=False)
    for t in ("<|im_end|>", "<|endoftext|>"):
        raw = raw.replace(t, "")
    think, answer, closed = split_reasoning(raw.strip(), finish)
    reasoning_tokens = len(tok(think, add_special_tokens=False)["input_ids"]) if think else 0
    return {
        "prompt": prompt, "answer": answer, "raw_output": raw.strip(),
        "generation_time_seconds": round(dt, 3), "tokens_generated": len(ids),
        # measured from the visible <think> text the model itself emitted (not an API-reported figure)
        "reasoning_tokens": reasoning_tokens, "total_tokens": n_prompt + len(ids),
        "finish_reason": finish, "think_block_closed": closed,
        "gpu_memory_used_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1) if runner.cuda else None,
    }


# %% [markdown]
# ## 6. Checkpointing

# %%
class Checkpoint:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"meta": {}, "generations": {}, "judge": {}, "pairwise": {}}
        if path.exists():
            try:
                self.data.update(json.loads(path.read_text(encoding="utf-8")))
                log.info("Resuming from checkpoint: %d generations, %d judgements, %d pairwise",
                         len(self.data["generations"]), len(self.data["judge"]), len(self.data["pairwise"]))
            except json.JSONDecodeError:
                bad = path.with_suffix(".corrupt.json")
                shutil.copy(path, bad)
                log.error("Checkpoint was malformed (copied to %s). Starting a fresh one.", bad)

    def save(self):
        atomic_write_json(self.path, self.data)

    def guard(self, bench_hash: str, specs: list[ModelSpec]):
        m = self.data["meta"]
        gen_cfg = {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "base": BASE_MODEL,
                   "system_prompt_sha": hashlib.sha256(SYSTEM_PROMPT.encode()).hexdigest()[:12],
                   "quantization": QUANTIZATION}
        if m.get("benchmark_sha256") and m["benchmark_sha256"] != bench_hash:
            raise SystemExit("Checkpoint belongs to a DIFFERENT benchmark hash. Delete evaluation_checkpoint.json "
                             "to start over (the benchmark must stay frozen across comparisons).")
        if m.get("gen_config") and m["gen_config"] != gen_cfg:
            raise SystemExit(f"Generation settings changed since the checkpoint was written:\n"
                             f"  old: {m['gen_config']}\n  new: {gen_cfg}\n"
                             "Delete evaluation_checkpoint.json to re-run under the new settings.")
        fps = m.setdefault("fingerprints", {})
        for s in specs:
            if s.path is None:
                continue
            fp = fingerprint(s.path)
            if fps.get(s.label) and fps[s.label] != fp:
                raise SystemExit(f"Weights for '{s.label}' changed since cached answers were produced. "
                                 "Delete evaluation_checkpoint.json (or use a new label in EXTRA_MODELS).")
            fps[s.label] = fp
        m["benchmark_sha256"], m["gen_config"] = bench_hash, gen_cfg
        self.save()


# %% [markdown]
# ## 7. Frozen CROP-100 benchmark

# %%
def load_and_freeze_benchmark(paths: Paths) -> tuple[list[dict], str]:
    frozen = paths.eval_dir / "crop_100_questions.json"
    hash_file = paths.eval_dir / "crop_100_hash.txt"
    src = None
    if frozen.exists():
        src = frozen
        for r in paths.roots:
            other = r / "crop_100_questions.json"
            if other.exists() and other.resolve() != frozen.resolve() and \
                    hashlib.sha256(other.read_bytes()).hexdigest() != hashlib.sha256(frozen.read_bytes()).hexdigest():
                raise SystemExit(f"{other} differs from the frozen benchmark {frozen}. The benchmark must not change "
                                 "between comparisons. Remove one of them deliberately if you really mean to re-freeze.")
    else:
        for r in paths.roots:
            if (r / "crop_100_questions.json").exists():
                src = r / "crop_100_questions.json"
                break
    if src is None:
        raise SystemExit(
            "crop_100_questions.json not found. Put your REAL CROP-100 file in the project folder.\n"
            "Accepted: a list of strings, or dicts with at least 'question'. For the full judge rubric also include\n"
            "gold_answer, gold_facts, acceptable_variations, category, difficulty, region. (The script does not invent\n"
            "benchmark questions or gold answers for you.)")
    raw_bytes = src.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    if hash_file.exists():
        old = hash_file.read_text(encoding="utf-8").split()[0]
        if old != digest:
            raise SystemExit(f"BENCHMARK CHANGED: frozen hash {old[:12]}... but {src} hashes to {digest[:12]}.... "
                             "Refusing to evaluate. Restore the original file.")
    else:
        hash_file.write_text(f"{digest}  crop_100_questions.json\n", encoding="utf-8")
        log.info("Benchmark frozen. SHA-256 = %s", digest)
    if src != frozen and not frozen.exists():
        shutil.copy(src, frozen)
    try:
        raw = json.loads(raw_bytes.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise SystemExit(f"{src} is not valid JSON: {e}")
    if isinstance(raw, dict):
        raw = next((v for v in raw.values() if isinstance(v, list)), raw)
    bench = []
    for i, q in enumerate(raw, 1):
        d = {"question": q} if isinstance(q, str) else dict(q)
        if not d.get("question"):
            raise SystemExit(f"benchmark item #{i} has no 'question'")
        bench.append({
            "id": d.get("id") or f"CROP_{i:03d}", "question": d["question"],
            "category": d.get("category") or "unspecified", "difficulty": d.get("difficulty") or "unspecified",
            "region": d.get("region") or "unspecified", "gold_answer": d.get("gold_answer") or "",
            "gold_facts": d.get("gold_facts") or [], "acceptable_variations": d.get("acceptable_variations") or []})
    if len(bench) != EXPECTED_QUESTIONS:
        raise SystemExit(f"benchmark has {len(bench)} questions, expected exactly {EXPECTED_QUESTIONS}.")
    if len({b["id"] for b in bench}) != len(bench):
        raise SystemExit("benchmark ids are not unique")
    n_gold = sum(1 for b in bench if b["gold_answer"])
    if n_gold < len(bench):
        log.warning("%d/%d questions have no gold_answer -> those are judged REFERENCE-FREE (weaker evidence).",
                    len(bench) - n_gold, len(bench))
    return bench, digest


# %% [markdown]
# ## 8. Embeddings, leakage audit, dataset quality, grouped split

# %%
def embed_groups(groups: list[list[str]]):
    """Returns (list_of_L2-normalised_matrices, backend_name). Falls back to TF-IDF if no sentence-transformers."""
    try:
        from sentence_transformers import SentenceTransformer
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        m = SentenceTransformer(EMBED_MODEL, device=dev)
        dim = m.get_sentence_embedding_dimension()
        outs = [m.encode(g, batch_size=128, normalize_embeddings=True, convert_to_numpy=True,
                         show_progress_bar=False) if g else np.zeros((0, dim), dtype=np.float32) for g in groups]
        del m
        free_cuda()
        return outs, f"sentence-transformers:{EMBED_MODEL}"
    except Exception as e:
        log.warning("sentence-transformers unavailable (%s: %s) -> TF-IDF cosine fallback. Thresholds are NOT "
                    "comparable to embedding cosine; install sentence-transformers for the real audit.",
                    type(e).__name__, e)
        from sklearn.feature_extraction.text import TfidfVectorizer
        allt = [t for g in groups for t in g]
        from scipy.sparse import csr_matrix
        vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1).fit(allt)
        return [vec.transform(g) if g else csr_matrix((0, len(vec.vocabulary_))) for g in groups], "tfidf-fallback"


def sim_matrix(a, b) -> np.ndarray:
    s = a @ b.T
    return s.toarray() if hasattr(s, "toarray") else np.asarray(s)


def load_training_rows(path: Path) -> list[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        raw = next(v for v in raw.values() if isinstance(v, list))
    return [r for r in raw if r.get("instruction") and r.get("response")]


def nb1_split(rows: list[dict]) -> tuple[list[int], list[int]]:
    """Reproduce the exact random 90/10 split (seed 42) used in agribot_nb1_local.py."""
    from datasets import Dataset
    ds = Dataset.from_list([{"idx": i, "instruction": r["instruction"], "response": r["response"],
                             "category": r.get("category", "")} for i, r in enumerate(rows)])
    sp = ds.train_test_split(test_size=0.10, seed=42, shuffle=True)
    return list(sp["train"]["idx"]), list(sp["test"]["idx"])


TEXT_KEYS = ("instruction", "prompt", "question", "input", "query")
SKIP_FILES = {"crop_100_questions.json", "log_history.json", "run_config.json", "adapter_config.json", "config.json",
              "trainer_state.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "special_tokens_map.json", "added_tokens.json", "vocab.json", "preprocessor_config.json"}


def discover_extra_sources(paths: Paths) -> list[tuple[str, list[str]]]:
    """Other JSON/JSONL files near the project (DPO pairs, older datasets...) whose prompts must not match CROP."""
    found, seen = [], set()
    for r in paths.roots:
        for dp, dns, fns in os.walk(r):
            dns[:] = [d for d in dns if d not in SKIP_DIRS and not d.startswith("checkpoint-")
                      and not d.startswith("weights_") and not d.startswith(".")]
            if Path(dp).resolve() != r.resolve() and len(Path(dp).relative_to(r).parts) > 3:
                dns[:] = []
            for fn in fns:
                p = Path(dp) / fn
                if fn in SKIP_FILES or p.suffix.lower() not in (".json", ".jsonl") or p.resolve() in seen:
                    continue
                if paths.dataset and p.resolve() == paths.dataset.resolve():
                    continue
                seen.add(p.resolve())
                try:
                    if p.stat().st_size > 300 * 2**20:
                        continue
                    txt = p.read_text(encoding="utf-8")
                    items = ([json.loads(l) for l in txt.splitlines() if l.strip()]
                             if p.suffix.lower() == ".jsonl" else json.loads(txt))
                    if isinstance(items, dict):
                        items = next((v for v in items.values() if isinstance(v, list)), [])
                    if not (isinstance(items, list) and items and isinstance(items[0], dict)):
                        continue
                    key = next((k for k in TEXT_KEYS if k in items[0]), None)
                    if key:
                        texts = [str(i.get(key, "")) for i in items if i.get(key)]
                        if texts:
                            found.append((str(p), texts))
                except Exception:
                    continue
    return found


def leakage_audit(bench: list[dict], rows: list[dict], split: Optional[tuple[list[int], list[int]]],
                  extra: list[tuple[str, list[str]]]) -> dict:
    src_names, src_texts = [], []
    train_set = set(split[0]) if split else set()
    for i, r in enumerate(rows):
        part = "dataset"
        if split:
            part = "dataset:train" if i in train_set else "dataset:validation"
        src_names.append(part)
        src_texts.append(r["instruction"])
    for fname, texts in extra:
        src_names += [fname] * len(texts)
        src_texts += texts
    # all benchmark phrasings (question + acceptable variations); a leak of ANY phrasing counts
    q_texts, q_owner = [], []
    for bi, b in enumerate(bench):
        for t in [b["question"]] + list(b["acceptable_variations"]):
            q_texts.append(t)
            q_owner.append(bi)
    norm_src = {}
    for n, (nm, t) in enumerate(zip(src_names, src_texts)):
        norm_src.setdefault(norm_text(t), n)
    exact = {}
    for t, bi in zip(q_texts, q_owner):
        n = norm_src.get(norm_text(t))
        if n is not None:
            exact[bi] = {"source": src_names[n], "text": src_texts[n]}
    gold = [b["gold_answer"] for b in bench if b["gold_answer"]]
    resp = [r["response"] for r in rows]
    (Q, S, G, R), backend = embed_groups([q_texts, src_texts, gold, resp])
    sims = sim_matrix(Q, S)
    best = np.full(len(bench), -1.0)
    best_src = [None] * len(bench)
    for k, bi in enumerate(q_owner):
        j = int(np.argmax(sims[k]))
        if sims[k, j] > best[bi]:
            best[bi], best_src[bi] = float(sims[k, j]), j
    ans_sim = {}
    if len(gold) and len(resp):
        gs = sim_matrix(G, R)
        gids = [b["id"] for b in bench if b["gold_answer"]]
        for k, qid in enumerate(gids):
            ans_sim[qid] = float(gs[k].max())
    flagged = []
    for bi, b in enumerate(bench):
        levels = [t for t in NEAR_THRESHOLDS if best[bi] >= t]
        is_exact = bi in exact
        if is_exact or levels:
            flagged.append({
                "id": b["id"], "exact_duplicate": is_exact, "max_similarity": round(float(best[bi]), 4),
                "highest_threshold_hit": max(levels) if levels else None,
                "nearest_source": src_names[best_src[bi]] if best_src[bi] is not None else None,
                "nearest_text": src_texts[best_src[bi]] if best_src[bi] is not None else None,
                "exact_match_source": exact.get(bi, {}).get("source")})
    flagged_ids = {f["id"] for f in flagged}
    rep = {
        "embedding_backend": backend, "thresholds": list(NEAR_THRESHOLDS),
        "total_benchmark_questions": len(bench),
        "training_sources": {"dataset_rows": len(rows), "extra_files": [{"file": f, "n": len(t)} for f, t in extra]},
        "exact_leakage_count": len(exact),
        "near_duplicate_counts": {f">={t}": int((best >= t).sum()) for t in NEAR_THRESHOLDS},
        "near_duplicate_count_at_0.90": int((best >= NEAR_THRESHOLDS[0]).sum()),
        "safe_count": len(bench) - len(flagged_ids), "flagged_questions": flagged,
        "flagged_ids": sorted(flagged_ids),
        "gold_answer_vs_training_responses_ge_0.90": sorted(q for q, s in ans_sim.items() if s >= 0.90),
        "note": ("Flagged questions are NOT removed. Scores are also reported on the leakage-clean subset."
                 if backend != "tfidf-fallback" else
                 "TF-IDF fallback used: similarity values are lexical and not comparable to the 0.90/0.95/0.97 "
                 "embedding thresholds. Install sentence-transformers and re-run for a valid audit.")}
    return rep


def length_stats(xs: list[int]) -> dict:
    a = np.array(xs) if xs else np.array([0])
    return {"mean": float(a.mean()), "median": float(np.median(a)), "min": int(a.min()), "max": int(a.max())}


def dataset_quality(rows: list[dict], split: Optional[tuple[list[int], list[int]]]) -> tuple[dict, Any]:
    from collections import Counter
    n = len(rows)
    q_norm = [norm_text(r["instruction"]) for r in rows]
    a_norm = [norm_text(r["response"]) for r in rows]
    rep: dict[str, Any] = {
        "n_rows": n,
        "exact_duplicate_questions": n - len(set(q_norm)),
        "exact_duplicate_answers": n - len(set(a_norm)),
        "category_counts": dict(Counter(r.get("category", "") for r in rows)),
        "region_focus_counts": dict(Counter(r.get("region_focus", "") for r in rows)),
        "answer_length_words": length_stats([len(r["response"].split()) for r in rows]),
        "question_length_words": length_stats([len(r["instruction"].split()) for r in rows]),
    }
    (X, A), backend = embed_groups([[r["instruction"] for r in rows], [r["response"] for r in rows]])
    rep["embedding_backend"] = backend
    for name, M in (("questions", X), ("answers", A)):
        s = sim_matrix(M, M)
        np.fill_diagonal(s, -1.0)
        rep[f"semantic_duplicate_{name}"] = {
            f">={t}": {"items_with_a_neighbour": int((s.max(1) >= t).sum()), "pairs": int((s >= t).sum() // 2)}
            for t in NEAR_THRESHOLDS}
        if name == "questions":
            sim_q = s.copy()
        del s
    if split:
        tr, va = split
        s = sim_matrix(X[va], X[tr]) if not hasattr(X, "tocsr") else sim_matrix(X[va], X[tr])
        mx = s.max(1)
        rep["train_validation_leakage_nb1_random_split"] = {
            "train": len(tr), "validation": len(va),
            **{f"val_items_with_train_neighbour>={t}": int((mx >= t).sum()) for t in NEAR_THRESHOLDS}}
    rep["warnings"] = []
    if rep["exact_duplicate_questions"]:
        rep["warnings"].append(f"{rep['exact_duplicate_questions']} exact duplicate questions (target: 0)")
    if rep["exact_duplicate_answers"]:
        rep["warnings"].append(f"{rep['exact_duplicate_answers']} exact duplicate answers (target: 0)")
    return rep, sim_q


def grouped_split(rows: list[dict], sim_q: np.ndarray, thr: float = 0.90, val_frac: float = 0.10) -> dict:
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    n = len(rows)
    adj = csr_matrix(sim_q >= thr)
    n_groups, labels = connected_components(adj, directed=False)
    sizes = np.bincount(labels)
    rng = np.random.default_rng(SEED)
    target = int(val_frac * n)
    max_group = max(2, int(0.05 * n))
    val_groups, count = set(), 0
    for g in rng.permutation(n_groups):
        if sizes[g] > max_group:
            continue
        if count >= target:
            break
        if count + sizes[g] > target * 1.1:
            continue
        val_groups.add(int(g))
        count += int(sizes[g])
    val_idx = [i for i in range(n) if int(labels[i]) in val_groups]
    train_idx = [i for i in range(n) if int(labels[i]) not in val_groups]
    return {"similarity_threshold": thr, "n_groups": int(n_groups), "singleton_groups": int((sizes == 1).sum()),
            "largest_group": int(sizes.max()), "groups_larger_than_5pct_kept_in_train": int((sizes > max_group).sum()),
            "train_ids": [rows[i].get("id", i) for i in train_idx], "val_ids": [rows[i].get("id", i) for i in val_idx],
            "note": "Whole similarity groups go to train OR validation. Not category-stratified."}


# %% [markdown]
# ## 9. Optional RAG retriever (separate mode, never mixed with model-only)

# %%
class Retriever:
    def __init__(self, docs_dir: Path, top_k: int):
        from sentence_transformers import SentenceTransformer
        self.k = top_k
        self.chunks: list[str] = []
        self.names: list[str] = []
        for p in sorted(docs_dir.rglob("*")):
            if p.suffix.lower() not in (".txt", ".md", ".json", ".jsonl") or not p.is_file():
                continue
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if p.suffix.lower() in (".json", ".jsonl"):
                try:
                    obj = [json.loads(l) for l in text.splitlines() if l.strip()] if p.suffix.lower() == ".jsonl" \
                        else json.loads(text)
                    text = "\n".join(json.dumps(o, ensure_ascii=False) if not isinstance(o, str) else o
                                     for o in (obj if isinstance(obj, list) else [obj]))
                except json.JSONDecodeError:
                    pass
            words = text.split()
            for s in range(0, max(1, len(words)), 160):
                ch = " ".join(words[s:s + 200])
                if ch.strip():
                    self.chunks.append(ch)
                    self.names.append(f"{p.name}#{s // 160}")
        if not self.chunks:
            raise RuntimeError(f"No readable documents under {docs_dir}")
        self.model = SentenceTransformer(EMBED_MODEL, device="cpu")   # CPU: keep VRAM for the LLM
        self.emb = self.model.encode(self.chunks, batch_size=64, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False)
        log.info("RAG index: %d chunks from %s", len(self.chunks), docs_dir)

    def retrieve(self, q: str) -> tuple[str, list[str]]:
        v = self.model.encode([q], normalize_embeddings=True, convert_to_numpy=True)[0]
        top = np.argsort(-(self.emb @ v))[: self.k]
        return "\n---\n".join(self.chunks[i] for i in top), [self.names[i] for i in top]


# %% [markdown]
# ## 10. Generation

# %%
def gen_key(mode: str, label: str, qid: str) -> str:
    return f"{mode}|{label}|{qid}"


def run_generation_for(runner: Runner, labels: list[str], bench: list[dict], ck: Checkpoint, mode: str,
                       retriever: Optional[Retriever]) -> list[str]:
    failures = []
    for label in labels:
        todo = [q for q in bench if gen_key(mode, label, q["id"]) not in ck.data["generations"]]
        if not todo:
            log.info("[%s/%s] all %d answers already in checkpoint", mode, label, len(bench))
            continue
        with runner.use(label):
            for q in tqdm(todo, desc=f"{mode}/{label}"):
                ctx, ctx_names = (retriever.retrieve(q["question"]) if retriever else (None, None))
                try:
                    rec = generate_one(runner, build_prompt(runner.tok, q["question"], ctx))
                except Exception as e:
                    import torch
                    oom = isinstance(e, torch.cuda.OutOfMemoryError)
                    free_cuda()
                    log.error("[%s/%s/%s] generation failed (%s): %s", mode, label, q["id"],
                              "CUDA OOM" if oom else type(e).__name__, e)
                    failures.append(gen_key(mode, label, q["id"]))
                    continue
                rec.update(question_id=q["id"], model=label, mode=mode)
                if ctx_names:
                    rec["retrieved_chunks"] = ctx_names
                ck.data["generations"][gen_key(mode, label, q["id"])] = rec
                ck.save()
    return failures


def generate_all(specs: list[ModelSpec], bench: list[dict], ck: Checkpoint, hw: dict, modes: list[str],
                 retriever: Optional[Retriever]) -> list[str]:
    failures: list[str] = []
    group = [s for s in specs if s.kind in ("base", "peft_adapter")]
    fulls = [s for s in specs if s.kind == "full_or_merged_hf_model"]

    def pending(labels, mode):
        return any(gen_key(mode, l, q["id"]) not in ck.data["generations"] for l in labels for q in bench)

    for mode in modes:
        labels = [s.label for s in group]
        if pending(labels, mode):
            runner = load_runner_for_group(group, hw)
            try:
                failures += run_generation_for(runner, labels, bench, ck, mode, retriever if mode == "rag" else None)
            finally:
                del runner
                free_cuda()
    for s in fulls:
        for mode in modes:
            if pending([s.label], mode):
                runner = load_runner_for_full(s, hw)
                try:
                    failures += run_generation_for(runner, [s.label], bench, ck, mode,
                                                   retriever if mode == "rag" else None)
                finally:
                    del runner
                    free_cuda()
    return failures


# %% [markdown]
# ## 11. Blinded LLM judges: Gemini + GPT-OSS with automatic failover, second pass, pairwise
#
# * Judge order comes from JUDGE_ORDER. The first live judge is the "primary".
# * Second pass (borderline answers): the OTHER judge is used when it is still live (independent second opinion);
#   otherwise the same judge re-grades with a stricter "skeptical reviewer" prompt.
# * Failover happens at question boundaries and all models' answers to a question are judged by the SAME primary
#   judge, so a quota switch never makes one model's answers be graded by a different judge than another's.
#   Which judge graded what is recorded in judge_scores.json and summarised in the report.

# %%
class QuotaExhausted(Exception):
    """A judge's daily quota is used up / it is unusable. Caught by the pool or by main()."""


def _flat(msg: str) -> str:
    return msg.lower().replace("_", "").replace(" ", "").replace("-", "")


class JudgeBackend:
    TRANSIENT = ("429", "500", "502", "503", "504", "resource_exhausted", "unavailable", "deadline", "timeout",
                 "timed out", "connection", "overloaded")
    DISABLE = ("apikeynotvalid", "invalidapikey", "permissiondenied", "unauthenticated", "http401", "http403",
               "http404", "notfound", "modelnotfound")

    def __init__(self, name: str, kind: str, model: str, key: str, base_url: Optional[str] = None,
                 min_interval: float = 7.0, extras: Optional[dict] = None):
        self.name, self.kind, self.model, self.key = name, kind, model, key
        self.base_url, self.min_interval = base_url, min_interval
        self.extras = dict(extras or {})
        self.use_extras = True
        self.exhausted = False
        self.exhausted_reason = ""
        self.calls = 0
        self._last = 0.0
        if kind == "gemini":
            from google import genai            # ImportError handled by build_pool
            from google.genai import types
            self.types = types
            self.client = genai.Client(api_key=key)

    def _mark(self, reason: str, detail: str):
        self.exhausted, self.exhausted_reason = True, reason
        log.warning("Judge '%s' (%s) disabled for this run: %s | %s", self.name, self.model, reason, detail[:200])

    def _call_once(self, prompt: str) -> str:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()
        self.calls += 1
        if self.kind == "gemini":
            resp = self.client.models.generate_content(
                model=self.model, contents=prompt,
                config=self.types.GenerateContentConfig(temperature=0.0, response_mime_type="application/json"))
            return resp.text or ""
        import requests
        body: dict[str, Any] = {"model": self.model, "temperature": 0,
                                "messages": [{"role": "user", "content": prompt}]}
        if self.use_extras:
            body["response_format"] = {"type": "json_object"}
            body.update(self.extras)
        url = self.base_url.rstrip("/") + "/chat/completions"
        hdr = {"Authorization": f"Bearer {self.key}"}
        r = requests.post(url, headers=hdr, json=body, timeout=180)
        if r.status_code == 400 and self.use_extras:      # host rejects json mode / reasoning params -> go plain
            self.use_extras = False
            log.warning("Judge '%s': HTTP 400 with JSON-mode/extras (%s). Retrying without them.", self.name, r.text[:150])
            body.pop("response_format", None)
            for k in self.extras:
                body.pop(k, None)
            r = requests.post(url, headers=hdr, json=body, timeout=180)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:300]}")
        return r.json()["choices"][0]["message"].get("content") or ""

    def ask_json(self, prompt: str, retries: int = 6) -> dict:
        last = None
        for attempt in range(retries):
            try:
                return parse_json_object(self._call_once(prompt))
            except (ValueError, KeyError) as e:             # malformed / empty JSON -> ask again
                last = e
                time.sleep(1.0)
            except Exception as e:
                last = e
                msg = str(e)
                f = _flat(msg)
                if any(h in f for h in ("perday", "dailylimit", "dailyquota")):
                    self._mark("daily quota exhausted", msg)
                    raise QuotaExhausted(f"{self.name}: {msg[:200]}") from e
                if any(h in f for h in self.DISABLE):
                    self._mark("rejected (bad key / no access / unknown model)", msg)
                    raise QuotaExhausted(f"{self.name}: {msg[:200]}") from e
                if not any(h in msg.lower() for h in self.TRANSIENT) and attempt >= 1:
                    break
                time.sleep(min(60, 2 ** attempt + random.random()))
        raise RuntimeError(f"{self.name}: judge call failed after retries: {last}")


class JudgePool:
    def __init__(self, backends: list[JudgeBackend]):
        self.backends = backends

    def live(self) -> list[JudgeBackend]:
        return [b for b in self.backends if not b.exhausted]

    def primary(self) -> JudgeBackend:
        live = self.live()
        if not live:
            raise QuotaExhausted("all judges exhausted/unavailable: " +
                                 "; ".join(f"{b.name}: {b.exhausted_reason}" for b in self.backends))
        return live[0]

    def other(self, primary: JudgeBackend) -> Optional[JudgeBackend]:
        return next((b for b in self.live() if b is not primary), None)

    def describe(self) -> str:
        return "Judges: " + " -> ".join(f"{b.name} ({b.model})" for b in self.backends)


def _env_first(names: list[str]) -> str:
    for n in names:
        if os.environ.get(n):
            return os.environ[n]
    return ""


def build_pool() -> Optional[JudgePool]:
    backends: list[JudgeBackend] = []
    for name in JUDGE_ORDER:
        try:
            if name == "gemini":
                key = GEMINI_API_KEY or _env_first(["GEMINI_API_KEY", "GOOGLE_API_KEY"])
                if not key:
                    log.warning("No Gemini key (GEMINI_API_KEY) -> Gemini judge skipped.")
                    continue
                backends.append(JudgeBackend("gemini", "gemini", GEMINI_MODEL, key,
                                             min_interval=GEMINI_MIN_INTERVAL_SECONDS))
            elif name == "gptoss":
                pre = GPTOSS_PRESETS.get(GPTOSS_PROVIDER)
                if pre is None:
                    raise SystemExit(f"GPTOSS_PROVIDER must be one of {list(GPTOSS_PRESETS)}")
                key = GPTOSS_API_KEY or _env_first(pre["env"])
                base = GPTOSS_BASE_URL or pre["base_url"]
                model = GPTOSS_MODEL or pre["model"]
                if not key:
                    log.warning("No GPT-OSS key (%s) -> GPT-OSS judge skipped.", " / ".join(pre["env"]))
                    continue
                if not base or not model:
                    raise SystemExit("GPTOSS_PROVIDER='custom' needs GPTOSS_BASE_URL and GPTOSS_MODEL.")
                backends.append(JudgeBackend("gptoss", "openai_compat", model, key, base_url=base,
                                             min_interval=GPTOSS_MIN_INTERVAL_SECONDS, extras=pre["extras"]))
        except ImportError as e:
            log.error("Judge '%s' unavailable (%s). For Gemini: pip install google-genai", name, e)
    return JudgePool(backends) if backends else None


def parse_json_object(text: str) -> dict:
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip(), flags=re.M).strip()
    m = re.search(r"\{.*\}", t, flags=re.S)
    if not m:
        raise ValueError("no JSON object in judge output")
    return json.loads(m.group(0))


def validate_judge(d: dict) -> dict:
    acc, comp = clamp(d.get("accuracy"), 0, 40), clamp(d.get("completeness"), 0, 25)
    rel, con = clamp(d.get("relevance"), 0, 20), clamp(d.get("conciseness"), 0, 15)
    out = {"accuracy": acc, "completeness": comp, "relevance": rel, "conciseness": con,
           "total": acc + comp + rel + con, "judge_reported_total": d.get("total"),
           "correct": bool(d.get("correct")), "critical_error": bool(d.get("critical_error")),
           "hallucination": bool(d.get("hallucination", False)),
           "confidence": clamp(d.get("confidence", 0.5), 0, 1), "reason": str(d.get("reason", ""))[:700]}
    if out["critical_error"]:
        out["correct"] = False                     # definition of "correct" forbids critical errors
    cat = d.get("error_category")
    out["error_category"] = None if out["correct"] else (cat if cat in ERROR_CATEGORIES else "FACTUAL_ERROR"
                                                         if out["critical_error"] else (cat or "UNCLASSIFIED"))
    return out


JUDGE_JSON_SPEC = ('{"accuracy": <0-40 int>, "completeness": <0-25 int>, "relevance": <0-20 int>, '
                   '"conciseness": <0-15 int>, "total": <sum>, "correct": <true|false>, '
                   '"critical_error": <true|false>, "hallucination": <true|false: invented facts, doses, dates, '
                   'diseases or sources>, "confidence": <0.0-1.0 your confidence in the correct flag>, '
                   '"error_category": <one of ' + json.dumps(ERROR_CATEGORIES) + ' or null if correct>, '
                   '"reason": "<2-3 sentences>"}')

JUDGE_RULES = """Scoring rubric (integers): accuracy 0-40 (correct agricultural information), completeness 0-25
(covers the important points of the gold answer), relevance 0-20 (directly answers the question asked),
conciseness 0-15 (clear and appropriately concise).

"correct" is true ONLY if ALL hold:
1. it states the key agricultural conclusion;
2. it contains no critical factual error (a mistake that would lead a farmer to a harmful or clearly wrong action);
3. it answers the question actually asked (for Pakistan-specific questions a generic global answer is NOT enough;
   if the correct advice depends on soil test / cultivar / location / irrigation / weather and the gold says so,
   an answer that ignores that dependency is incomplete);
4. different wording is fine - do NOT require word-for-word match with the gold answer;
5. extra correct information is allowed;
6. a short answer can still be fully correct.
Do not reward length. Do not invent facts. Only judge the final answer text shown to you."""


def fmt_gold(b: dict) -> str:
    if not b["gold_answer"]:
        return "(none provided - judge from established agronomy knowledge and say so in the reason)"
    s = b["gold_answer"]
    if b["gold_facts"]:
        s += "\nKEY GOLD FACTS:\n" + "\n".join(f"- {f}" for f in b["gold_facts"])
    if b["acceptable_variations"]:
        s += "\nACCEPTABLE VARIATIONS:\n" + "\n".join(f"- {f}" for f in b["acceptable_variations"])
    return s


def judge_prompt(b: dict, answer: str, skeptical: bool) -> str:
    role = ("You are a SKEPTICAL second reviewer. Actively look for critical errors, unsupported recommendations and "
            "hallucinated specifics before awarding 'correct'." if skeptical else
            "You are an expert agronomist grading ONE anonymous answer.")
    return (f"{role}\nYou do not know who or what wrote the answer.\n\nQUESTION:\n{b['question']}\n\n"
            f"GOLD REFERENCE:\n{fmt_gold(b)}\n\nANSWER TO GRADE:\n{answer}\n\n{JUDGE_RULES}\n\n"
            f"Return ONLY a JSON object exactly like: {JUDGE_JSON_SPEC}")


def needs_second_pass(j: dict) -> bool:
    return (SECOND_JUDGE_BAND[0] <= j["total"] <= SECOND_JUDGE_BAND[1]) or \
           (j["correct"] and j["confidence"] < SECOND_JUDGE_MIN_CONF)


def combine(j1: dict, j2: Optional[dict]) -> dict:
    if j2 is None:
        return {**{k: j1[k] for k in ("accuracy", "completeness", "relevance", "conciseness", "total", "correct",
                                       "critical_error", "hallucination", "error_category", "confidence", "reason")},
                "judge_disagreement": False, "manual_review": False, "judged_by": "judge1",
                "correct_judge1": j1["correct"]}
    dims = {k: (j1[k] + j2[k]) / 2 for k in ("accuracy", "completeness", "relevance", "conciseness")}
    total = sum(dims.values())
    disagree = (j1["correct"] != j2["correct"]) or abs(j1["total"] - j2["total"]) > DISAGREE_SCORE_DELTA
    correct = j1["correct"] and j2["correct"]          # strict: both judges must agree it is correct
    cat = None if correct else next((j["error_category"] for j in (j1, j2) if not j["correct"]), "UNCLASSIFIED")
    return {**dims, "total": total, "correct": correct,
            "critical_error": j1["critical_error"] or j2["critical_error"],
            "hallucination": j1["hallucination"] or j2["hallucination"] or cat == "HALLUCINATION",
            "error_category": cat, "confidence": min(j1["confidence"], j2["confidence"]),
            "reason": f"J1: {j1['reason']} | J2: {j2['reason']}", "judge_disagreement": disagree,
            "manual_review": disagree, "judged_by": "judge1+judge2", "correct_judge1": j1["correct"]}


def judge_one(pool: JudgePool, primary: JudgeBackend, b: dict, gen: dict) -> dict:
    if not gen["answer"].strip():
        return {"accuracy": 0, "completeness": 0, "relevance": 0, "conciseness": 0, "total": 0, "correct": False,
                "critical_error": False, "hallucination": False, "error_category": NO_ANSWER_CATEGORY,
                "confidence": 1.0, "reason": "No final answer was produced (generation ended inside the thinking "
                                             "block or was empty).", "judge_disagreement": False,
                "manual_review": False, "judged_by": "auto", "correct_judge1": False, "judge1_backend": "auto",
                "judge1_model": None, "judge2_backend": None, "judge2_model": None,
                "second_judge_independent": None}
    j1 = validate_judge(primary.ask_json(judge_prompt(b, gen["answer"], False)))
    j2, second, independent = None, None, False
    if RUN_SECOND_JUDGE and needs_second_pass(j1):
        second = pool.other(primary)
        independent = second is not None
        if second is None:
            second = primary
        try:
            j2 = validate_judge(second.ask_json(judge_prompt(b, gen["answer"], True)))
        except QuotaExhausted:
            if second is primary:
                raise
            second, independent = primary, False         # the other judge just died -> same judge, skeptical prompt
            j2 = validate_judge(primary.ask_json(judge_prompt(b, gen["answer"], True)))
    rec = combine(j1, j2)
    rec.update(judge1_backend=primary.name, judge1_model=primary.model,
               judge2_backend=second.name if j2 else None, judge2_model=second.model if j2 else None,
               second_judge_independent=independent if j2 else None)
    return rec


def run_judging(bench: list[dict], labels: list[str], ck: Checkpoint, mode: str, pool: JudgePool) -> None:
    for q in tqdm(bench, desc=f"judge {mode}"):
        qid = q["id"]
        todo = [l for l in labels if gen_key(mode, l, qid) in ck.data["generations"]
                and gen_key(mode, l, qid) not in ck.data["judge"]]
        if not todo:
            continue
        while True:                                   # same primary judge for every model on this question
            primary = pool.primary()                  # raises QuotaExhausted when no judge is left
            staged: dict[str, dict] = {}
            try:
                for label in todo:
                    try:
                        staged[label] = judge_one(pool, primary, q, ck.data["generations"][gen_key(mode, label, qid)])
                    except QuotaExhausted:
                        raise
                    except Exception as e:
                        log.error("judge failed for %s/%s/%s: %s (will retry on next run)", mode, label, qid, e)
                break
            except QuotaExhausted as e:
                log.warning("Judge '%s' out on question %s (%s). Switching judge and re-grading this question for "
                            "ALL models so they share one judge.", primary.name, qid, str(e)[:120])
        for label, rec in staged.items():
            rec.update(question_id=qid, model=label, mode=mode)
            ck.data["judge"][gen_key(mode, label, qid)] = rec
        ck.save()


def pair_prompt(b: dict, x: str, y: str) -> str:
    return (f"You are an expert agronomist comparing two anonymous answers.\n\nQUESTION:\n{b['question']}\n\n"
            f"GOLD REFERENCE:\n{fmt_gold(b)}\n\nANSWER_X:\n{x}\n\nANSWER_Y:\n{y}\n\n"
            "Pick the better answer on correctness first, then completeness and relevance, then concision. "
            "Prefer the answer without critical errors. Say 'tie' only if they are genuinely equal in quality. "
            'Return ONLY JSON: {"winner": "X" | "Y" | "tie", "reason": "<1-2 sentences>"}')


def pair_one(primary: JudgeBackend, q: dict, a: str, b_: str, ga: dict, gb: dict, key: str, mode: str) -> dict:
    rng = random.Random(f"{SEED}|{key}")
    swap = rng.random() < 0.5                          # randomise which answer is shown first
    x_label, y_label = (b_, a) if swap else (a, b_)
    ea, eb = not ga["answer"].strip(), not gb["answer"].strip()
    backend = "auto"
    if ea and eb:
        winner, reason = "tie", "both answers empty"
    elif ea or eb:
        winner, reason = (b_ if ea else a), "only one answer was non-empty"
    else:
        xa = (gb if swap else ga)["answer"]
        ya = (ga if swap else gb)["answer"]
        r = primary.ask_json(pair_prompt(q, xa, ya))
        backend = primary.name
        w = str(r.get("winner", "tie")).strip().upper()
        winner = x_label if w == "X" else y_label if w == "Y" else "tie"
        reason = str(r.get("reason", ""))[:400]
    return {"mode": mode, "pair": [a, b_], "question_id": q["id"], "winner": winner, "reason": reason,
            "position": {"X": x_label, "Y": y_label}, "judge_backend": backend}


def run_pairwise(bench: list[dict], labels: list[str], ck: Checkpoint, mode: str, pool: JudgePool) -> None:
    main = [l for l in ("base", "fine_tuned", "moa") if l in labels]
    pairs = list(itertools.combinations(main, 2))
    for q in tqdm(bench, desc=f"pairwise {mode}"):
        todo = []
        for a, b_ in pairs:
            key = f"{mode}|{a}__vs__{b_}|{q['id']}"
            ga = ck.data["generations"].get(gen_key(mode, a, q["id"]))
            gb = ck.data["generations"].get(gen_key(mode, b_, q["id"]))
            if key not in ck.data["pairwise"] and ga and gb:
                todo.append((a, b_, ga, gb, key))
        if not todo:
            continue
        while True:
            primary = pool.primary()
            staged: dict[str, dict] = {}
            try:
                for a, b_, ga, gb, key in todo:
                    try:
                        staged[key] = pair_one(primary, q, a, b_, ga, gb, key, mode)
                    except QuotaExhausted:
                        raise
                    except Exception as e:
                        log.error("pairwise failed %s: %s", key, e)
                break
            except QuotaExhausted as e:
                log.warning("Judge '%s' out during pairwise on %s (%s). Switching judge.", primary.name, q["id"],
                            str(e)[:120])
        ck.data["pairwise"].update(staged)
        ck.save()


# %% [markdown]
# ## 12. Statistics, error analysis, diagnostics

# %%
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def boot_ci(x: np.ndarray, B: int = 10000) -> tuple[float, float]:
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def score_stats(x: np.ndarray) -> dict:
    lo, hi = boot_ci(x)
    return {"mean": float(x.mean()), "median": float(np.median(x)), "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
            "min": float(x.min()), "max": float(x.max()), "ci95_low": lo, "ci95_high": hi}


def mcnemar_exact(a: np.ndarray, b: np.ndarray) -> dict:
    only_a = int((a & ~b).sum())
    only_b = int((~a & b).sum())
    n = only_a + only_b
    if n == 0:
        p = 1.0
    else:
        try:
            from scipy.stats import binomtest
            p = float(binomtest(min(only_a, only_b), n, 0.5).pvalue)
        except ImportError:
            from scipy.stats import binom_test
            p = float(binom_test(min(only_a, only_b), n, 0.5))
    return {"a_correct_b_wrong": only_a, "a_wrong_b_correct": only_b, "p_value": p}


def paired_boot_diff(a: np.ndarray, b: np.ndarray, B: int = 10000) -> dict:
    rng = np.random.default_rng(SEED)
    d = a.astype(float) - b.astype(float)
    idx = rng.integers(0, len(d), (B, len(d)))
    m = d[idx].mean(axis=1)
    return {"mean_diff": float(d.mean()), "ci95_low": float(np.percentile(m, 2.5)),
            "ci95_high": float(np.percentile(m, 97.5))}


def collect_mode(bench, labels, ck: Checkpoint, mode: str) -> tuple[dict, list[str]]:
    """Per-model records restricted to the questions judged for EVERY model (paired design)."""
    have = [{q["id"] for q in bench if gen_key(mode, l, q["id"]) in ck.data["judge"]} for l in labels]
    common = set.intersection(*have) if have else set()
    ids = [q["id"] for q in bench if q["id"] in common]
    if len(ids) < len(bench):
        log.warning("[%s] only %d/%d questions judged for all models; statistics use those.", mode, len(ids), len(bench))
    return {l: [ck.data["judge"][gen_key(mode, l, i)] for i in ids] for l in labels}, ids


def analyse_mode(bench, labels, ck: Checkpoint, mode: str, leak: dict) -> dict:
    by_id = {b["id"]: b for b in bench}
    recs, ids = collect_mode(bench, labels, ck, mode)
    n = len(ids)
    out: dict[str, Any] = {"mode": mode, "n_questions": n, "models": {}, "pairs": {}, "pairwise": {}}
    if n == 0:
        return out
    flagged = set(leak.get("flagged_ids", []))
    clean_idx = [k for k, i in enumerate(ids) if i not in flagged]
    corr = {l: np.array([r["correct"] for r in recs[l]], dtype=bool) for l in labels}
    for l in labels:
        c, tot = corr[l], np.array([r["total"] for r in recs[l]], dtype=float)
        k = int(c.sum())
        gens = [ck.data["generations"][gen_key(mode, l, i)] for i in ids]
        cat_acc = {}
        for cat in sorted({by_id[i]["category"] for i in ids}):
            m = np.array([by_id[i]["category"] == cat for i in ids])
            cat_acc[cat] = {"correct": int(c[m].sum()), "n": int(m.sum()), "accuracy": float(c[m].mean())}
        diff_acc = {}
        for d in sorted({by_id[i]["difficulty"] for i in ids}):
            m = np.array([by_id[i]["difficulty"] == d for i in ids])
            diff_acc[d] = {"correct": int(c[m].sum()), "n": int(m.sum()), "accuracy": float(c[m].mean())}
        w_lo, w_hi = wilson(k, n)
        b_lo, b_hi = boot_ci(c.astype(float))
        out["models"][l] = {
            "correct": k, "n": n, "accuracy": k / n, "wilson_ci95": [w_lo, w_hi], "bootstrap_ci95": [b_lo, b_hi],
            "correct_judge1_only": int(sum(r["correct_judge1"] for r in recs[l])),
            "clean_subset": {"n": len(clean_idx), "correct": int(c[clean_idx].sum()) if clean_idx else 0,
                             "accuracy": float(c[clean_idx].mean()) if clean_idx else None},
            "judge_total": score_stats(tot),
            "dimension_means": {d: float(np.mean([r[d] for r in recs[l]]))
                                for d in ("accuracy", "completeness", "relevance", "conciseness")},
            "hallucination_count": int(sum(r["hallucination"] for r in recs[l])),
            "critical_error_count": int(sum(r["critical_error"] for r in recs[l])),
            "judge_disagreements": int(sum(r["judge_disagreement"] for r in recs[l])),
            "judge_mix": dict(__import__("collections").Counter(r.get("judge1_backend", "?") for r in recs[l])),
            "second_pass_independent": int(sum(1 for r in recs[l] if r.get("second_judge_independent") is True)),
            "second_pass_same_judge": int(sum(1 for r in recs[l] if r.get("second_judge_independent") is False)),
            "truncated_no_answer": int(sum(1 for g in gens if not g["answer"].strip())),
            "finish_reason_length": int(sum(1 for g in gens if g["finish_reason"] == "length")),
            "mean_generation_seconds": float(np.mean([g["generation_time_seconds"] for g in gens])),
            "mean_tokens_generated": float(np.mean([g["tokens_generated"] for g in gens])),
            "mean_reasoning_tokens": float(np.mean([g["reasoning_tokens"] or 0 for g in gens])),
            "per_category": cat_acc, "per_difficulty": diff_acc}
    for a, b_ in itertools.combinations(labels, 2):
        out["pairs"][f"{a}__vs__{b_}"] = {
            "accuracy_diff_pp": float((corr[a].mean() - corr[b_].mean()) * 100),
            "mcnemar": mcnemar_exact(corr[a], corr[b_]),
            "bootstrap_acc_diff": paired_boot_diff(corr[a], corr[b_]),
            "bootstrap_score_diff": paired_boot_diff(np.array([r["total"] for r in recs[a]]),
                                                     np.array([r["total"] for r in recs[b_]]))}
    for key, v in ck.data["pairwise"].items():
        if not key.startswith(mode + "|") or v["question_id"] not in set(ids):
            continue
        pk = f"{v['pair'][0]}__vs__{v['pair'][1]}"
        d = out["pairwise"].setdefault(pk, {v["pair"][0]: 0, v["pair"][1]: 0, "tie": 0})
        d[v["winner"]] += 1
    for pk, d in out["pairwise"].items():
        a, b_ = pk.split("__vs__")
        nt = d[a] + d[b_]
        if nt:
            from scipy.stats import binomtest
            d["sign_test_p"] = float(binomtest(min(d[a], d[b_]), nt, 0.5).pvalue)
    # error analysis
    rows = []
    for l in labels:
        for q, r in zip(ids, recs[l]):
            if not r["correct"]:
                b = by_id[q]
                rows.append({"mode": mode, "model": l, "question_id": q, "crop_category": b["category"],
                             "difficulty": b["difficulty"], "region": b["region"],
                             "error_category": r["error_category"] or "UNCLASSIFIED",
                             "critical_error": r["critical_error"], "hallucination": r["hallucination"],
                             "judge_total": r["total"], "manual_review": r["manual_review"],
                             "reason": r["reason"]})
    out["error_rows"] = rows
    out["error_distribution"] = {}
    for l in labels:
        errs = [r["error_category"] for r in rows if r["model"] == l]
        out["error_distribution"][l] = {c: {"count": errs.count(c), "percentage": 100 * errs.count(c) / len(errs)}
                                        for c in sorted(set(errs))} if errs else {}
    out["question_ids"] = ids
    return out


INTERVENTIONS = {
    "FACTUAL_ERROR": "Add verified Q&A (FAO/PARC/provincial extension sources) on the failing concepts; remove templated "
                     "answers that state facts generically.",
    "INCOMPLETE": "Add examples whose answers cover all key points (causes + diagnosis + action + dependencies).",
    "WRONG_CROP": "Add crop-contrast examples (same symptom/practice, different crop) so crop identity is respected.",
    "WRONG_REGION": "Add Pakistan province-specific examples (Punjab / Sindh / KP / Balochistan) with contrasting answers.",
    "WRONG_SEASON": "Add Rabi-vs-Kharif and sowing/harvest-window reasoning examples that name the season explicitly.",
    "WRONG_DISEASE": "Add symptom-comparison / differential-diagnosis pairs (disease vs disease, disease vs deficiency).",
    "WRONG_SOIL_CHEMISTRY": "Add contrastive soil examples: salinity vs sodicity, pH vs nutrient availability, "
                            "EC/SAR interpretation, gypsum logic.",
    "UNSUPPORTED_RECOMMENDATION": "Add examples that state dependencies (soil test, cultivar, local advisory) instead of "
                                  "giving rates/doses; consider DPO pairs penalising unsupported specifics.",
    "HALLUCINATION": "Add 'what to check / defer to extension' examples; consider DPO with a hallucinated answer as "
                     "the rejected sample.",
    "QUESTION_MISUNDERSTANDING": "Add varied phrasings and scenario-style questions so the model answers what is asked.",
    NO_ANSWER_CATEGORY: "Generation never left the thinking block within the token limit. Either raise MAX_NEW_TOKENS for "
                        "ALL models, or train/prompt the model to close <think> immediately (the fine-tune does).",
}


def training_diagnostics(log_candidates: list[Path], summary: dict) -> list[str]:
    msgs = []
    path = next((p for p in log_candidates if p.exists()), None)
    if not path:
        return ["No training log_history.json found next to the Notebook-1 weights; loss-vs-CROP divergence not checked."]
    try:
        logs = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return [f"Could not parse {path}"]
    tr = [(l["step"], l["loss"]) for l in logs if "loss" in l]
    ev = [(l["step"], l["eval_loss"]) for l in logs if "eval_loss" in l]
    ta = [(l["step"], l["mean_token_accuracy"]) for l in logs if "mean_token_accuracy" in l]
    msgs.append(f"Training log: {path}")
    loss_down = False
    if len(tr) >= 8:
        q = max(1, len(tr) // 4)
        first, last = np.mean([x[1] for x in tr[:q]]), np.mean([x[1] for x in tr[-q:]])
        loss_down = last < first
        msgs.append(f"Train loss {first:.4f} -> {last:.4f}")
    if ev:
        best = min(ev, key=lambda x: x[1])
        msgs.append(f"Validation loss: best {best[1]:.4f} at step {best[0]}, last {ev[-1][1]:.4f} at step {ev[-1][0]}")
        if ev[-1][0] > best[0] and ev[-1][1] > best[1] * 1.02:
            msgs.append("WARNING: validation loss rose after its minimum -> overfitting risk; more epochs will likely "
                        "not help.")
    mo = summary.get("model_only", {}).get("models", {})
    if "base" in mo and "fine_tuned" in mo:
        ft, bs = mo["fine_tuned"]["correct"], mo["base"]["correct"]
        pair = summary["model_only"]["pairs"].get("base__vs__fine_tuned", {})
        p = pair.get("mcnemar", {}).get("p_value", 1.0)
        if loss_down and ft <= bs:
            msgs.append(f"WARNING: training loss went DOWN but CROP did not improve (fine_tuned {ft} vs base {bs}). "
                        "Likely overfitting / poor generalisation / template learning.")
        if ta and ta[-1][1] > ta[0][1] + 0.02 and (ft <= bs or p > 0.05):
            msgs.append("WARNING: token accuracy rose but CROP is unchanged or not significantly better "
                        f"(McNemar p={p:.3f}). Treat as possible memorisation / template learning.")
    return msgs


# %% [markdown]
# ## 13. Charts, tables, report

# %%
def make_charts(summary: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    modes = [m for m in ("model_only", "rag") if m in summary and summary[m].get("models")]
    if not modes:
        return
    # accuracy
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = list(summary[modes[0]]["models"])
    w = 0.8 / len(modes)
    for mi, mode in enumerate(modes):
        accs = [summary[mode]["models"][l]["accuracy"] * 100 for l in labels]
        err = [[a - summary[mode]["models"][l]["wilson_ci95"][0] * 100 for a, l in zip(accs, labels)],
               [summary[mode]["models"][l]["wilson_ci95"][1] * 100 - a for a, l in zip(accs, labels)]]
        xs = np.arange(len(labels)) + mi * w
        ax.bar(xs, accs, w, yerr=err, capsize=4, label=mode)
        for x, a in zip(xs, accs):
            ax.text(x, a + 1, f"{a:.0f}", ha="center", fontsize=9)
    ax.axhline(TARGET, color="red", ls="--", lw=1, label=f"target {TARGET}")
    ax.set_xticks(np.arange(len(labels)) + w * (len(modes) - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CROP-100 correct (%)")
    ax.set_ylim(0, 105)
    ax.set_title("CROP-100 accuracy (95% Wilson CI)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "crop_accuracy.png", dpi=150)
    plt.close(fig)
    # judge scores (primary mode)
    S = summary[modes[0]]["models"]
    dims = ["accuracy", "completeness", "relevance", "conciseness"]
    mx = {"accuracy": 40, "completeness": 25, "relevance": 20, "conciseness": 15}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [3, 1.3]})
    ww = 0.8 / len(S)
    for i, l in enumerate(S):
        axes[0].bar(np.arange(4) + i * ww, [S[l]["dimension_means"][d] / mx[d] * 100 for d in dims], ww, label=l)
    axes[0].set_xticks(np.arange(4) + ww * (len(S) - 1) / 2)
    axes[0].set_xticklabels([f"{d}\n(max {mx[d]})" for d in dims])
    axes[0].set_ylabel("% of dimension maximum")
    axes[0].set_ylim(0, 105)
    axes[0].legend()
    axes[0].set_title("Judge dimensions")
    axes[1].bar(list(S), [S[l]["judge_total"]["mean"] for l in S],
                yerr=[[S[l]["judge_total"]["mean"] - S[l]["judge_total"]["ci95_low"] for l in S],
                      [S[l]["judge_total"]["ci95_high"] - S[l]["judge_total"]["mean"] for l in S]], capsize=4)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("Overall score (0-100)")
    fig.tight_layout()
    fig.savefig(out_dir / "judge_scores.png", dpi=150)
    plt.close(fig)
    # per-category accuracy
    cats = sorted({c for l in S for c in S[l]["per_category"]})
    fig, ax = plt.subplots(figsize=(max(8, 1.6 * len(cats)), 5))
    for i, l in enumerate(S):
        ax.bar(np.arange(len(cats)) + i * ww, [S[l]["per_category"].get(c, {"accuracy": 0})["accuracy"] * 100
                                               for c in cats], ww, label=l)
    ax.set_xticks(np.arange(len(cats)) + ww * (len(S) - 1) / 2)
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_ylabel("correct (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.set_title(f"Per-category CROP accuracy ({modes[0]})")
    fig.tight_layout()
    fig.savefig(out_dir / "category_accuracy.png", dpi=150)
    plt.close(fig)
    # error distribution
    ed = summary[modes[0]]["error_distribution"]
    ecats = sorted({c for l in ed for c in ed[l]})
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * max(1, len(ecats))), 5))
    for i, l in enumerate(ed):
        ax.bar(np.arange(len(ecats)) + i * ww, [ed[l].get(c, {"count": 0})["count"] for c in ecats], ww, label=l)
    ax.set_xticks(np.arange(len(ecats)) + ww * (len(ed) - 1) / 2)
    ax.set_xticklabels(ecats, rotation=30, ha="right")
    ax.set_ylabel("number of incorrect answers")
    ax.legend()
    ax.set_title(f"Error categories ({modes[0]})")
    fig.tight_layout()
    fig.savefig(out_dir / "error_distribution.png", dpi=150)
    plt.close(fig)


def write_tables(summary: dict, out_dir: Path) -> None:
    import pandas as pd
    srows, crow, erow = [], [], []
    for mode in ("model_only", "rag"):
        s = summary.get(mode)
        if not s or not s.get("models"):
            continue
        for l, m in s["models"].items():
            srows.append({"mode": mode, "model": l, "correct": m["correct"], "n": m["n"],
                          "accuracy_pct": 100 * m["accuracy"], "wilson_lo_pct": 100 * m["wilson_ci95"][0],
                          "wilson_hi_pct": 100 * m["wilson_ci95"][1], "judge_mean": m["judge_total"]["mean"],
                          "judge_median": m["judge_total"]["median"], "judge_std": m["judge_total"]["std"],
                          "judge_min": m["judge_total"]["min"], "judge_max": m["judge_total"]["max"],
                          **{f"mean_{d}": v for d, v in m["dimension_means"].items()},
                          "hallucinations": m["hallucination_count"], "critical_errors": m["critical_error_count"],
                          "judge_disagreements": m["judge_disagreements"],
                          "no_answer_truncated": m["truncated_no_answer"],
                          "correct_judge1_only": m["correct_judge1_only"],
                          "clean_subset_n": m["clean_subset"]["n"], "clean_subset_correct": m["clean_subset"]["correct"],
                          "mean_gen_seconds": m["mean_generation_seconds"], "mean_tokens": m["mean_tokens_generated"]})
            for c, v in m["per_category"].items():
                crow.append({"mode": mode, "model": l, "category": c, **v})
        erow += s["error_rows"]
    pd.DataFrame(srows).to_csv(out_dir / "crop_summary.csv", index=False)
    pd.DataFrame(crow).to_csv(out_dir / "category_results.csv", index=False)
    pd.DataFrame(erow).to_csv(out_dir / "error_analysis.csv", index=False)


def improvement_report(summary: dict) -> dict:
    s = summary.get("model_only")
    if not s or not s.get("models"):
        return {}
    rep = {}
    for l in s["models"]:
        if l == "base":
            continue
        errs = [r for r in s["error_rows"] if r["model"] == l]
        by_cat, by_err = {}, {}
        for r in errs:
            by_cat[r["crop_category"]] = by_cat.get(r["crop_category"], 0) + 1
            by_err[r["error_category"]] = by_err.get(r["error_category"], 0) + 1
        top = sorted(by_err.items(), key=lambda x: -x[1])
        rep[l] = {"errors": len(errs), "errors_by_crop_category": dict(sorted(by_cat.items(), key=lambda x: -x[1])),
                  "errors_by_error_type": dict(top),
                  "most_common_failure": top[0][0] if top else None,
                  "recommended_interventions": [{"error_type": e, "count": c, "action": INTERVENTIONS.get(e, "Inspect manually.")}
                                                for e, c in top],
                  "rules": ["Write NEW questions that test the same underlying concept; never copy CROP questions, "
                            "gold answers or paraphrases of them into training data.",
                            "Re-run the leakage audit on the new dataset before training.",
                            "Change one thing per experiment and re-run the frozen CROP-100."]}
    return rep


def pick_best(models: dict) -> tuple[str, list[str]]:
    """Best by CROP correct count, then judge mean. Returns (label, labels tied on correct count)."""
    ranked = sorted(models.items(), key=lambda kv: (kv[1]["correct"], kv[1]["judge_total"]["mean"]), reverse=True)
    top = ranked[0]
    tied = [l for l, m in ranked[1:] if m["correct"] == top[1]["correct"]]
    return top[0], tied


def render_markdown(R: dict) -> str:
    L: list[str] = []
    L.append("# AgriBot CROP-100 Evaluation Report\n")
    L.append(f"Generated: {R['generated_at']}  |  Benchmark SHA-256: `{R['benchmark_sha256']}`\n")
    if R.get("smoke"):
        L.append(f"> **SMOKE TEST on the first {R['smoke']} questions only - NOT a valid CROP-100 result.**\n")
    L.append("## Executive summary\n")
    mo = R["results"].get("model_only", {})
    if mo.get("models"):
        best_l, tied = pick_best(mo["models"])
        best = (best_l, mo["models"][best_l])
        for l, m in mo["models"].items():
            L.append(f"- **{l}**: {m['correct']}/{m['n']} correct ({100 * m['accuracy']:.1f}%), judge mean "
                     f"{m['judge_total']['mean']:.1f}/100")
        gap = max(0, TARGET - best[1]["correct"])
        clean = best[1]["clean_subset"]
        L.append(f"\nPrevious best (your figure): {PREVIOUS_BEST}/100  |  Target: {TARGET}/100  |  "
                 f"Best measured: {best[1]['correct']}/100 ({best[0]})  |  Gap: {gap} question(s)")
        achieved = best[1]["correct"] >= TARGET
        tie_txt = f" (tied on CROP correct count with: {', '.join(tied)}; ranked first on judge mean)" if tied else ""
        L.append(f"\n**Best model: {best[0]}{tie_txt}. Target {TARGET}/100: "
                 f"{'ACHIEVED' if achieved else 'NOT achieved'}** (model-only mode).")
        if clean["accuracy"] is not None and R["leakage"]["flagged_ids"]:
            L.append(f"Leakage-clean subset: {clean['correct']}/{clean['n']} "
                     f"({100 * clean['accuracy']:.1f}%) - {len(R['leakage']['flagged_ids'])} flagged question(s) excluded.")
        if achieved and R["leakage"]["flagged_ids"]:
            L.append("**Caution: some benchmark questions were flagged by the leakage audit; treat the headline number "
                     "with care until the flags are resolved.**")
    else:
        L.append("_No judged results yet (generation only, or judge not run)._")
    if R.get("skipped_models"):
        L.append("\nSkipped models: " + "; ".join(f"{s['label']} ({s['reason']})" for s in R["skipped_models"]))
    L.append("\n## Dataset (CROP-100)\n")
    L.append(f"- Questions: {R['benchmark_stats']['n']}")
    L.append(f"- Categories: {R['benchmark_stats']['categories']}")
    L.append(f"- Difficulty: {R['benchmark_stats']['difficulty']}")
    L.append(f"- Questions with gold answers: {R['benchmark_stats']['with_gold']}")
    lk = R["leakage"]
    L.append(f"\n### Leakage audit ({lk['embedding_backend']})\n")
    L.append(f"- exact leakage: {lk['exact_leakage_count']}  |  near-duplicates {lk['near_duplicate_counts']}  |  "
             f"safe: {lk['safe_count']}/{lk['total_benchmark_questions']}")
    L.append(f"- flagged: {lk['flagged_ids'] or 'none'}")
    L.append(f"- note: {lk['note']}")
    L.append("\n## Model configuration\n")
    for m in R["models"]:
        L.append(f"- **{m['label']}** ({m['role']}): kind={m['kind']}, path={m['path']}, quantization={R['config']['quantization']}, "
                 f"dtype={R['hardware'].get('dtype')}, GPU={R['hardware'].get('gpu')}")
    L.append(f"- Generation: do_sample=False (greedy), max_new_tokens={R['config']['max_new_tokens']}, identical system "
             "prompt for all models; judged on the final answer only (text after </think>).")
    for mode, s in R["results"].items():
        if not s.get("models"):
            continue
        L.append(f"\n## Results - {mode}\n")
        L.append("| Model | Correct | Acc % (95% Wilson) | Judge mean | Acc/40 | Comp/25 | Rel/20 | Conc/15 | Halluc. | "
                 "No-answer (truncated) | Disagreements |")
        L.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for l, m in s["models"].items():
            d = m["dimension_means"]
            L.append(f"| {l} | {m['correct']}/{m['n']} | {100 * m['accuracy']:.1f} "
                     f"({100 * m['wilson_ci95'][0]:.0f}-{100 * m['wilson_ci95'][1]:.0f}) | {m['judge_total']['mean']:.1f} | "
                     f"{d['accuracy']:.1f} | {d['completeness']:.1f} | {d['relevance']:.1f} | {d['conciseness']:.1f} | "
                     f"{m['hallucination_count']} | {m['truncated_no_answer']} | {m['judge_disagreements']} |")
        L.append("\nJudge that graded each model's answers (first pass): " +
                 "; ".join(f"{l}: {m['judge_mix']}" for l, m in s["models"].items()) +
                 "  |  second pass independent/same-judge: " +
                 "; ".join(f"{l}: {m['second_pass_independent']}/{m['second_pass_same_judge']}"
                           for l, m in s["models"].items()))
        L.append("\n**Per-category accuracy**\n")
        cats = sorted({c for m in s["models"].values() for c in m["per_category"]})
        L.append("| Category | " + " | ".join(s["models"]) + " |")
        L.append("|---|" + "---:|" * len(s["models"]))
        for c in cats:
            L.append(f"| {c} | " + " | ".join(f"{m['per_category'].get(c, {}).get('correct', '-')}/"
                                               f"{m['per_category'].get(c, {}).get('n', '-')}"
                                               for m in s["models"].values()) + " |")
        L.append("\n**Statistical comparison (paired)**\n")
        for pk, p in s["pairs"].items():
            a, b = pk.split("__vs__")
            mc, bd = p["mcnemar"], p["bootstrap_acc_diff"]
            L.append(f"- {a} vs {b}: {p['accuracy_diff_pp']:+.1f} pp; McNemar exact p={mc['p_value']:.4f} "
                     f"({a} only right: {mc['a_correct_b_wrong']}, {b} only right: {mc['a_wrong_b_correct']}); "
                     f"bootstrap diff {100 * bd['mean_diff']:+.1f} pp (95% CI {100 * bd['ci95_low']:+.1f} to "
                     f"{100 * bd['ci95_high']:+.1f})")
        if s["pairwise"]:
            L.append("\n**Pairwise judge preference**\n")
            for pk, d in s["pairwise"].items():
                L.append(f"- {pk.replace('__vs__', ' vs ')}: {json.dumps({k: v for k, v in d.items()})}")
        L.append("\n**Error analysis (incorrect answers)**\n")
        for l, dist in s["error_distribution"].items():
            L.append(f"- {l}: " + (", ".join(f"{c} {v['count']} ({v['percentage']:.0f}%)" for c, v in dist.items())
                                   or "no errors"))
    L.append("\n## Experiment table\n")
    L.append("| Experiment | CROP correct | Accuracy | Judge score | Hallucination |")
    L.append("|---|---:|---:|---:|---:|")
    for l, m in mo.get("models", {}).items():
        L.append(f"| {l} | {m['correct']}/{m['n']} | {100 * m['accuracy']:.0f}% | {m['judge_total']['mean']:.1f} | "
                 f"{m['hallucination_count']} |")
    L.append("\n## Training diagnostics\n")
    for msg in R["training_diagnostics"]:
        L.append(f"- {msg}")
    L.append("\n## Recommended next intervention\n")
    for l, rep in R["improvement"].items():
        L.append(f"**{l}** - {rep['errors']} errors; by CROP category {rep['errors_by_crop_category']}")
        for it in rep["recommended_interventions"]:
            L.append(f"- {it['error_type']} x{it['count']}: {it['action']}")
        for rule in rep["rules"]:
            L.append(f"- _{rule}_")
    L.append("\n### Priority order\n")
    for step in R["priority_notes"]:
        L.append(f"- {step}")
    if mo.get("models"):
        best_l, tied = pick_best(mo["models"])
        bm = mo["models"][best_l]
        who = f"**{best_l}** ({bm['correct']}/{bm['n']})" if not tied else \
            f"no clear winner: {best_l} and {', '.join(tied)} tie at {bm['correct']}/{bm['n']}"
        L.append(f"\n## Recommendation\n\nBy independent CROP accuracy (not loss or token accuracy): {who}. Prefer a model "
                 "for AgriBot only if its advantage is supported by the paired tests above.")
    return "\n".join(L) + "\n"


# %% [markdown]
# ## 14. Main pipeline

# %%
def parse_cli(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="AgriBot CROP-100 evaluation (Base vs Fine-Tuned vs MoA)")
    ap.add_argument("--smoke", type=int, default=0, help="quick pipeline test on the first N questions (separate output folder)")
    ap.add_argument("--generate-only", action="store_true", help="generate answers, skip judging")
    ap.add_argument("--judge-only", action="store_true", help="judge answers already in the checkpoint, no GPU needed")
    ap.add_argument("--no-pairwise", action="store_true", help="skip pairwise comparisons (saves judge quota)")
    ap.add_argument("--no-second-judge", action="store_true", help="grade each answer once")
    ap.add_argument("--project-dir"), ap.add_argument("--nb1"), ap.add_argument("--nb2")
    ap.add_argument("--max-new-tokens", type=int)
    args, _ = ap.parse_known_args(argv)
    g = globals()
    if args.generate_only:
        g["RUN_JUDGE"] = False
    if args.judge_only:
        g["RUN_GENERATION"] = False
    if args.no_pairwise:
        g["RUN_PAIRWISE"] = False
    if args.no_second_judge:
        g["RUN_SECOND_JUDGE"] = False
    if args.project_dir:
        g["PROJECT_DIR"] = args.project_dir
    if args.nb1:
        g["NOTEBOOK_1_WEIGHTS"] = args.nb1
    if args.nb2:
        g["NOTEBOOK_2_WEIGHTS"] = args.nb2
    if args.max_new_tokens:
        g["MAX_NEW_TOKENS"] = args.max_new_tokens
    return args


def load_env_files(paths: Paths) -> None:
    """Optional .env (KEY=VALUE lines) in the project or script folder; real environment variables win."""
    for d in (paths.project, SCRIPT_DIR):
        f = d / ".env"
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def main(argv=None):
    args = parse_cli(argv)
    random.seed(SEED)
    np.random.seed(SEED)
    paths = Paths()
    load_env_files(paths)
    hw = hardware_report()
    print_hardware(hw, paths)
    if RUN_GENERATION and not hw.get("cuda_available") and not ALLOW_CPU:
        raise SystemExit("No CUDA GPU detected by PyTorch. Check `nvidia-smi`, install a CUDA build of PyTorch "
                         "(pytorch.org), or set ALLOW_CPU=True (very slow).")
    if paths.dataset is None:
        raise SystemExit(f"Dataset {DATASET_FILENAME} not found under {[str(r) for r in paths.roots]}")

    # 1) frozen benchmark
    bench, bench_hash = load_and_freeze_benchmark(paths)
    log.info("Benchmark: %d questions, SHA-256 %s", len(bench), bench_hash)
    if args.smoke:
        bench = bench[:args.smoke]
        paths.eval_dir = paths.eval_dir / f"_smoke_{args.smoke}"
        paths.eval_dir.mkdir(parents=True, exist_ok=True)
        log.warning("SMOKE TEST: first %d questions only; outputs go to %s", args.smoke, paths.eval_dir)

    # 2) dataset checks + leakage audit (before any model touches the GPU)
    rows = load_training_rows(paths.dataset)
    try:
        split = nb1_split(rows)
    except Exception as e:
        log.warning("Could not reproduce the NB1 split (%s); train/validation leakage check skipped.", e)
        split = None
    extra = discover_extra_sources(paths)
    log.info("Extra training-like sources scanned for leakage: %s", [f for f, _ in extra] or "none")
    leak = leakage_audit(bench, rows, split, extra)
    atomic_write_json(paths.eval_dir / "leakage_report.json", leak)
    print(f"\nLEAKAGE AUDIT: total={leak['total_benchmark_questions']} exact={leak['exact_leakage_count']} "
          f"near>=0.90={leak['near_duplicate_counts']} safe={leak['safe_count']} flagged={len(leak['flagged_ids'])}")
    if leak["exact_leakage_count"]:
        print("  !! Exact duplicates between CROP and training data. See leakage_report.json. Nothing was removed.")
    dq = None
    if RUN_DATASET_TOOLS:
        dq, sim_q = dataset_quality(rows, split)
        atomic_write_json(paths.eval_dir / "dataset_quality_report.json", dq)
        atomic_write_json(paths.eval_dir / "grouped_split_suggestion.json", grouped_split(rows, sim_q))
        del sim_q
        for w in dq["warnings"]:
            print("  DATASET WARNING:", w)

    # 3) models
    specs, skipped = discover_models(paths)
    print_model_report(specs, skipped)
    labels = [s.label for s in specs]
    modes = ["model_only"] + (["rag"] if RAG_DOCS_DIR else [])
    retriever = Retriever(paths.resolve_user_path(RAG_DOCS_DIR), RAG_TOP_K) if RAG_DOCS_DIR else None

    ck = Checkpoint(paths.eval_dir / "evaluation_checkpoint.json")
    ck.guard(bench_hash, specs)

    # 4) generation
    failures: list[str] = []
    if RUN_GENERATION:
        failures = generate_all(specs, bench, ck, hw, modes, retriever)
        if failures:
            log.error("%d generations failed (see log). Re-run the script to retry them.", len(failures))

    # 5) judging (Gemini first, GPT-OSS takes over when Gemini's quota ends - or the reverse per JUDGE_ORDER)
    judged = False
    if RUN_JUDGE:
        pool = build_pool()
        if pool is None:
            log.error("No judge API key found (GEMINI_API_KEY / GROQ_API_KEY ...). Generation is saved; "
                      "set a key and re-run to judge.")
        else:
            print(pool.describe())
            try:
                for mode in modes:
                    have = [l for l in labels if any(gen_key(mode, l, q["id"]) in ck.data["generations"] for q in bench)]
                    run_judging(bench, have, ck, mode, pool)
                    if RUN_PAIRWISE:
                        run_pairwise(bench, have, ck, mode, pool)
            except QuotaExhausted as e:
                log.error("ALL judges are out of quota/unavailable (%s). Progress is saved: run the script again "
                          "after the quotas reset (Gemini resets midnight Pacific), or add another key. Writing "
                          "partial results.", e)
            print("Judge calls this run: " + ", ".join(f"{b.name}={b.calls}" for b in pool.backends))
            judged = True

    # 6) outputs that exist even without judging
    outputs = sorted(ck.data["generations"].values(), key=lambda r: (r["mode"], r["model"], r["question_id"]))
    atomic_write_json(paths.eval_dir / "model_outputs.json", outputs)
    if not judged and not ck.data["judge"]:
        print("\nGeneration finished; judging not run. Outputs saved in", paths.eval_dir)
        return
    atomic_write_json(paths.eval_dir / "judge_scores.json",
                      sorted(ck.data["judge"].values(), key=lambda r: (r["mode"], r["model"], r["question_id"])))
    atomic_write_json(paths.eval_dir / "pairwise_results.json", list(ck.data["pairwise"].values()))

    # 7) analysis
    results = {mode: analyse_mode(bench, labels, ck, mode, leak) for mode in modes
               if any(k.startswith(mode + "|") for k in ck.data["judge"])}
    nb1 = next((s for s in specs if s.label == "fine_tuned"), None)
    logc = [nb1.path / "log_history.json", nb1.path.parent / "log_history.json"] if nb1 and nb1.path else []
    diag = training_diagnostics(logc, results)
    improvement = improvement_report(results)
    from collections import Counter
    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "benchmark_sha256": bench_hash, "smoke": args.smoke,
        "config": {"max_new_tokens": MAX_NEW_TOKENS, "do_sample": False, "quantization": QUANTIZATION,
                   "base_model": BASE_MODEL, "judge_order": JUDGE_ORDER, "gemini_model": GEMINI_MODEL,
                   "gptoss_provider": GPTOSS_PROVIDER,
                   "gptoss_model": GPTOSS_MODEL or GPTOSS_PRESETS.get(GPTOSS_PROVIDER, {}).get("model"),
                   "target": TARGET, "seed": SEED},
        "hardware": hw,
        "models": [{"label": s.label, "role": s.role, "kind": s.kind, "path": str(s.path) if s.path else BASE_MODEL,
                    "info": s.info} for s in specs],
        "skipped_models": skipped,
        "benchmark_stats": {"n": len(bench), "categories": dict(Counter(b["category"] for b in bench)),
                            "difficulty": dict(Counter(b["difficulty"] for b in bench)),
                            "with_gold": sum(1 for b in bench if b["gold_answer"])},
        "leakage": leak, "dataset_quality": dq, "results": results, "training_diagnostics": diag,
        "improvement": improvement, "failed_generations": failures,
        "priority_notes": [
            "P1 dataset leakage/duplication: " + (
                "; ".join(dq["warnings"]) if dq and dq["warnings"] else "no exact duplicates in training data")
            + f"; CROP flagged: {len(leak['flagged_ids'])}",
            "P2 independent CROP-100: this run (frozen hash above).",
            "P3/P4 failure analysis -> targeted error-correction data: see interventions above.",
            "P5 new SFT/QLoRA run, P6 DPO/DoRA only if SFT plateaus, P7 MoA: add each as a label in EXTRA_MODELS and "
            "re-run; every technique is scored on the same frozen CROP-100."]}
    atomic_write_json(paths.eval_dir / "evaluation_report.json", report)
    write_tables(results, paths.eval_dir)
    make_charts(results, paths.eval_dir)
    (paths.eval_dir / "evaluation_report.md").write_text(render_markdown(report), encoding="utf-8")
    print("\n" + render_markdown(report))
    print("All reports saved to:", paths.eval_dir)


# %%
if __name__ == "__main__":
    main()
