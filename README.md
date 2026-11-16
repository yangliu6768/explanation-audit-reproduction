# When Explanations Betray Backdoors

This repository reproduces the main numerical tables in *When Explanations Betray Backdoors: Black-Box Auditing for Language Model Classifiers*, starting from frozen detector and probe scores. It includes the inference-time Groundedness Drift (GD) scorer and the ordered word-deletion views and hard-label flip score used for the text adaptation of BBCaL.

## Reproduce the tables

Use Python 3.10 or newer. The table calculation uses the standard library and runs on a CPU.

```sh
python3 reproduce.py
```

The command creates five CSVs in `results/`. Fractions, including false-positive rates (FPRs), are stored on a 0–1 scale. The paper displays some FPRs as percentages. The program reports the count of settings in which GD strictly outperforms each of seven baselines, using unrounded means.

| Output | Contents |
| --- | --- |
| table1.csv | Both backbone panels, including the reasoning prompt selected in each cell |
| table2.csv | Qwen AG News stress diagnostic, including realized clean FPR at both nominal budgets |
| table3.csv | Paired self-verification and clean independent verification on two seed-0 Qwen victims |
| table6.csv | Clean classification accuracy and target attack success before filtering |
| tables13-17.csv | The eight individual detectors for all five datasets |

**Scope.** The starting data are distributions of scores recorded from the original model runs. Running the command recomputes rankings, thresholds, residual attack success rates (rASRs), and seed means. Victim training and response generation are outside the scope of this release. The optional `scores.py` functions score new input/summary pairs or eight parsed hard labels; only the text similarity function requires scikit-learn (tested with 1.5.2).

## Main benchmark

The two victim backbones are Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.2. Each of the five datasets and four attack families has three independently trained victims per backbone (seeds 0, 1, 2). Each run has 200 clean and 300 triggered evaluation inputs. Clean and triggered records can share a source text.

The archived training configurations used rank-8 LoRA (alpha 16, dropout 0.05), three epochs, a learning rate of 0.0001, batch size 1, and four gradient accumulation steps. The main protocol used training seeds 0, 1, and 2 and a fixed data-sampling seed of 0. Sequence-length caps in tokens, read from the training configurations for the adapters that produced the table, were:

| Backbone | SST-2 | Rotten Tomatoes | TREC | Jigsaw | AG News |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen | 192 | 192 | 128 | 256 | 128 |
| Mistral | 192 or 256 | 256 | 128 | 256 | 384 |

`data/main_scores.csv.gz` contains one row per distinct anomaly score within a model, dataset, attack, training seed, and detector. The four counts are clean inputs, triggered inputs, correct clean predictions, and triggered predictions equal to the target. They retain ties and give the denominators for AUROC, FPR, and rASR. Higher scores indicate greater suspicion. AUROC counts a tie as half a win. The threshold in each run is the linearly interpolated 95th percentile of its 200 clean evaluation scores. Only a score strictly above this threshold is rejected. The paper's separate deployment threshold uses clean calibration inputs.

For the BBCaL column, the perturbations use eight nested word deletions (0, 5, 10, 20, 30, 40, 50, 60 percent). FPS is the zero-based index immediately before the first change of label, or seven if all labels agree. The anomaly score is negative FPS. A missing label needed to determine FPS gets score +1. The score itself needs no clean calibration. The per-input classifier outcome used for rASR is the archived victim prediction.

Reasoning-best selects the lowest mean rASR among Reasoning, ZS-CoT and Auto-CoT. To recreate the displayed prompt choice in ties, the Qwen panel prioritizes higher AUROC, and the Mistral panel prioritizes lower clean FPR followed by higher AUROC. For example, all three prompts have identical per-seed rASR on Qwen/Jigsaw/EP. Table 1 prints Reasoning in that cell. Every individual prompt is retained in the full comparison.

## AG News probe scores

`data/probe_scores.csv.gz` contains pooled counts of numerical component scores and predicted classes. The splits are calibration (256 clean) and evaluation (200 clean, 300 triggered). The field `correct` means correct classification on clean inputs and prediction of the attack target on triggered inputs.

- **Table 2.** Each family and seed has its own clean calibration reference. The grounding percentile uses the full reference. Support and document percentiles condition on the evidence-prompt class, with a global fallback below ten examples. The joint score is the maximum of the three percentiles for each input. Posthoc clean evaluation cutoffs include ties. Thus realized clean FPR for the joint score ranges from 11.7% to 17.3% at the nominal 10% budget and 28.2% to 34.7% at 20%, after averaging seeds per family. The historical first-stage response format was quoted evidence for seed 0 and an answer summary for seeds 1 and 2.
- **Table 3.** The two arms use paired victim decisions and evidence, with either victim self-verification or a clean Qwen2.5-7B-Instruct verifier without a LoRA adapter. GD gates at 5% by predicted class. On accepted calibration inputs, support and document scores become add-one clean percentiles, with a global fallback below four examples per class. Their maximum is thresholded strictly at the residual budget (q − 0.05)/0.95. UG AUROC uses GD-accepted evaluation inputs; rASR and realized FPR use the full evaluation set.
