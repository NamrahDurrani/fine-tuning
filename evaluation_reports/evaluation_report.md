# AgriBot CROP-100 Evaluation Report

Generated: 2026-10-07 18:58:06  |  Benchmark SHA-256: `98368208e07f84d13b62013c0ca00736ffecdea71deb65dfab15d2603cf336f4`

## Executive summary

- **base**: 29/82 correct (35.4%), judge mean 45.1/100
- **fine_tuned**: 46/82 correct (56.1%), judge mean 67.1/100

Previous best (your figure): 88/100  |  Target: 95/100  |  Best measured: 46/100 (fine_tuned)  |  Gap: 49 question(s)

**Best model: fine_tuned. Target 95/100: NOT achieved** (model-only mode).

Skipped models: moa (Model C - Optimized MoA (Notebook 2): no weights found (looked for weights_nb2 under ['/home/rapidsai/Desktop/fine-tuning', '/home/rapidsai/Desktop/fine-tuning/AgriBot_Project']))

## Dataset (CROP-100)

- Questions: 100
- Categories: {'Unknown': 100}
- Difficulty: {'unspecified': 100}
- Questions with gold answers: 0

### Leakage audit (sentence-transformers:sentence-transformers/all-MiniLM-L6-v2)

- exact leakage: 0  |  near-duplicates {'>=0.9': 0, '>=0.95': 0, '>=0.97': 0}  |  safe: 100/100
- flagged: none
- note: Flagged questions are NOT removed. Scores are also reported on the leakage-clean subset.

## Model configuration

- **base** (Model A - Base): kind=base, path=Qwen/Qwen3-4B-Thinking-2507, quantization=4bit, dtype=bfloat16, GPU=NVIDIA GeForce RTX 3090
- **fine_tuned** (Model B - Fine-Tuned (Notebook 1)): kind=peft_adapter, path=/home/rapidsai/Desktop/fine-tuning/AgriBot_Project/weights_nb1/final_adapter, quantization=4bit, dtype=bfloat16, GPU=NVIDIA GeForce RTX 3090
- Generation: do_sample=False (greedy), max_new_tokens=1024, identical system prompt for all models; judged on the final answer only (text after </think>).

## Results - model_only

| Model | Correct | Acc % (95% Wilson) | Judge mean | Acc/40 | Comp/25 | Rel/20 | Conc/15 | Halluc. | No-answer (truncated) | Disagreements |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base | 29/82 | 35.4 (26-46) | 45.1 | 18.5 | 9.2 | 9.4 | 7.9 | 5 | 28 | 3 |
| fine_tuned | 46/82 | 56.1 (45-66) | 67.1 | 27.4 | 13.1 | 13.3 | 13.2 | 0 | 0 | 1 |

Judge that graded each model's answers (first pass): base: {'gptoss': 54, 'auto': 28}; fine_tuned: {'gptoss': 82}  |  second pass independent/same-judge: base: 0/6; fine_tuned: 0/7

**Per-category accuracy**

| Category | base | fine_tuned |
|---|---:|---:|
| Unknown | 29/82 | 46/82 |

**Statistical comparison (paired)**

- base vs fine_tuned: -20.7 pp; McNemar exact p=0.0076 (base only right: 10, fine_tuned only right: 27); bootstrap diff -20.7 pp (95% CI -34.1 to -7.3)

**Error analysis (incorrect answers)**

- base: FACTUAL_ERROR 6 (11%), HALLUCINATION 1 (2%), INCOMPLETE 17 (32%), NO_ANSWER_TRUNCATED 28 (53%), QUESTION_MISUNDERSTANDING 1 (2%)
- fine_tuned: FACTUAL_ERROR 5 (14%), INCOMPLETE 30 (83%), QUESTION_MISUNDERSTANDING 1 (3%)

## Experiment table

| Experiment | CROP correct | Accuracy | Judge score | Hallucination |
|---|---:|---:|---:|---:|
| base | 29/82 | 35% | 45.1 | 5 |
| fine_tuned | 46/82 | 56% | 67.1 | 0 |

## Training diagnostics

- Training log: /home/rapidsai/Desktop/fine-tuning/AgriBot_Project/weights_nb1/log_history.json
- Train loss 0.5204 -> 0.0687
- Validation loss: best 0.0698 at step 800, last 0.0698 at step 846

## Recommended next intervention

**fine_tuned** - 36 errors; by CROP category {'Unknown': 36}
- INCOMPLETE x30: Add examples whose answers cover all key points (causes + diagnosis + action + dependencies).
- FACTUAL_ERROR x5: Add verified Q&A (FAO/PARC/provincial extension sources) on the failing concepts; remove templated answers that state facts generically.
- QUESTION_MISUNDERSTANDING x1: Add varied phrasings and scenario-style questions so the model answers what is asked.
- _Write NEW questions that test the same underlying concept; never copy CROP questions, gold answers or paraphrases of them into training data._
- _Re-run the leakage audit on the new dataset before training._
- _Change one thing per experiment and re-run the frozen CROP-100._

### Priority order

- P1 dataset leakage/duplication: no exact duplicates in training data; CROP flagged: 0
- P2 independent CROP-100: this run (frozen hash above).
- P3/P4 failure analysis -> targeted error-correction data: see interventions above.
- P5 new SFT/QLoRA run, P6 DPO/DoRA only if SFT plateaus, P7 MoA: add each as a label in EXTRA_MODELS and re-run; every technique is scored on the same frozen CROP-100.

## Recommendation

By independent CROP accuracy (not loss or token accuracy): **fine_tuned** (46/82). Prefer a model for AgriBot only if its advantage is supported by the paired tests above.
