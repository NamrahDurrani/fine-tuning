# AgriBot CROP-100 Evaluation Report

Generated: 2026-10-07 16:58:13  |  Benchmark SHA-256: `98368208e07f84d13b62013c0ca00736ffecdea71deb65dfab15d2603cf336f4`

> **SMOKE TEST on the first 3 questions only - NOT a valid CROP-100 result.**

## Executive summary

- **base**: 2/3 correct (66.7%), judge mean 69.7/100
- **fine_tuned**: 2/3 correct (66.7%), judge mean 67.7/100

Previous best (your figure): 88/100  |  Target: 95/100  |  Best measured: 2/100 (base)  |  Gap: 93 question(s)

**Best model: base (tied on CROP correct count with: fine_tuned; ranked first on judge mean). Target 95/100: NOT achieved** (model-only mode).

Skipped models: moa (Model C - Optimized MoA (Notebook 2): no weights found (looked for weights_nb2 under ['/home/rapidsai/Desktop/fine-tuning', '/home/rapidsai/Desktop/fine-tuning/AgriBot_Project']))

## Dataset (CROP-100)

- Questions: 3
- Categories: {'Unknown': 3}
- Difficulty: {'unspecified': 3}
- Questions with gold answers: 0

### Leakage audit (sentence-transformers:sentence-transformers/all-MiniLM-L6-v2)

- exact leakage: 0  |  near-duplicates {'>=0.9': 0, '>=0.95': 0, '>=0.97': 0}  |  safe: 3/3
- flagged: none
- note: Flagged questions are NOT removed. Scores are also reported on the leakage-clean subset.

## Model configuration

- **base** (Model A - Base): kind=base, path=Qwen/Qwen3-4B-Thinking-2507, quantization=4bit, dtype=bfloat16, GPU=NVIDIA GeForce RTX 3090
- **fine_tuned** (Model B - Fine-Tuned (Notebook 1)): kind=peft_adapter, path=/home/rapidsai/Desktop/fine-tuning/AgriBot_Project/weights_nb1/final_adapter, quantization=4bit, dtype=bfloat16, GPU=NVIDIA GeForce RTX 3090
- Generation: do_sample=False (greedy), max_new_tokens=1024, identical system prompt for all models; judged on the final answer only (text after </think>).

## Results - model_only

| Model | Correct | Acc % (95% Wilson) | Judge mean | Acc/40 | Comp/25 | Rel/20 | Conc/15 | Halluc. | No-answer (truncated) | Disagreements |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 2/3 | 66.7 (21-94) | 69.7 | 26.7 | 15.0 | 15.0 | 13.0 | 2 | 0 | 0 |
| fine_tuned | 2/3 | 66.7 (21-94) | 67.7 | 30.0 | 11.5 | 14.2 | 12.0 | 0 | 0 | 0 |

Judge that graded each model's answers (first pass): base: {'gptoss': 3}; fine_tuned: {'gptoss': 3}  |  second pass independent/same-judge: base: 0/0; fine_tuned: 0/1

**Per-category accuracy**

| Category | base | fine_tuned |
|---|---:|---:|
| Unknown | 2/3 | 2/3 |

**Statistical comparison (paired)**

- base vs fine_tuned: +0.0 pp; McNemar exact p=1.0000 (base only right: 1, fine_tuned only right: 1); bootstrap diff +0.0 pp (95% CI -100.0 to +100.0)

**Pairwise judge preference**

- base vs fine_tuned: {"base": 3, "fine_tuned": 0, "tie": 0, "sign_test_p": 0.25}

**Error analysis (incorrect answers)**

- base: FACTUAL_ERROR 1 (100%)
- fine_tuned: INCOMPLETE 1 (100%)

## Experiment table

| Experiment | CROP correct | Accuracy | Judge score | Hallucination |
|---|---:|---:|---:|---:|
| base | 2/3 | 67% | 69.7 | 2 |
| fine_tuned | 2/3 | 67% | 67.7 | 0 |

## Training diagnostics

- Training log: /home/rapidsai/Desktop/fine-tuning/AgriBot_Project/weights_nb1/log_history.json
- Train loss 0.5204 -> 0.0687
- Validation loss: best 0.0698 at step 800, last 0.0698 at step 846
- WARNING: training loss went DOWN but CROP did not improve (fine_tuned 2 vs base 2). Likely overfitting / poor generalisation / template learning.
- WARNING: token accuracy rose but CROP is unchanged or not significantly better (McNemar p=1.000). Treat as possible memorisation / template learning.

## Recommended next intervention

**fine_tuned** - 1 errors; by CROP category {'Unknown': 1}
- INCOMPLETE x1: Add examples whose answers cover all key points (causes + diagnosis + action + dependencies).
- _Write NEW questions that test the same underlying concept; never copy CROP questions, gold answers or paraphrases of them into training data._
- _Re-run the leakage audit on the new dataset before training._
- _Change one thing per experiment and re-run the frozen CROP-100._

### Priority order

- P1 dataset leakage/duplication: no exact duplicates in training data; CROP flagged: 0
- P2 independent CROP-100: this run (frozen hash above).
- P3/P4 failure analysis -> targeted error-correction data: see interventions above.
- P5 new SFT/QLoRA run, P6 DPO/DoRA only if SFT plateaus, P7 MoA: add each as a label in EXTRA_MODELS and re-run; every technique is scored on the same frozen CROP-100.

## Recommendation

By independent CROP accuracy (not loss or token accuracy): no clear winner: base and fine_tuned tie at 2/3. Prefer a model for AgriBot only if its advantage is supported by the paired tests above.
