# figures — the exact code behind each research question's figure

One folder per research question: the script, the `materials/` it reads (the outputs of `experiments/analysis`,
where the figure is not drawn straight from the result files) and the `output/` it writes. All self-contained apart
from `common.py` (palette, style, loaders, bootstrap).

```
python rq02_plausibility_tiers/rq02_plausibility_tiers.py --results <results root> --out <dir>
```
`--results` defaults to `COT3D_RESULTS` or `results/` at the repository root (`scripts/fetch_results.py`); `--out`
defaults to the script's own `output/`. Every script writes `<name>.pdf`, `<name>.png`
(300 dpi) and, where there are computed numbers, `<name>.json` beside them.
Intervals are per-fact percentile bootstraps (B=10,000, seed 0) unless stated.

| RQ | script | inputs | outputs |
|---|---|---|---|
| RQ1 | `rq01_dose_and_leak/rq01_dose_and_leak.py` | A1 (base, false_1k/3k/10k), A3 (false_3k) | `rq01_dose_response`, `rq01_leak_by_domain`, `rq01_model_key` |
| RQ2 | `rq02_plausibility_tiers/rq02_plausibility_tiers.py` | A1 (base, false_3k) | `rq02_tier_bootstrap` |
| RQ3 | `rq03_generalisation/rq03_generalisation.py` | A1 (base, false_3k), B4 (base, false_3k) | `rq03_delta_forest` (main), `rq03_heatmap_appendix` |
| RQ5 | `rq05_cot_use/rq05_cot_use.py` (`--rq5 results/rq05_cot_use`) | rq5_slot.py result files, three twins | main `rq05_override_symmetry` |
| RQ6 | `rq06_read_the_cot/rq06_read_the_cot.py` | results/rq06_read_the_cot/rq6 (oracle3t, blind3t), results/rq06_read_the_cot/rq6_traces | main `rq06_readers` |
| RQ7 | `rq07_grpo/rq07_grpo.py` (`--rq7 results/rq07_grpo`) | eval_judge.json, reward_log.jsonl and checkpoint_eval_judge.json of the two GRPO runs | main `rq07_prompt_conditional`; appendix `rq07_checkpoint_curve` (fixed-prompt curve over the saved checkpoints, verifier and judge), `rq07_reward_curves` |
| RQ8 | `rq08_monitor/rq08_monitor.py` | figures/rq08_monitor/materials/rq8_monitor.json (experiments/analysis/rq8_monitor_effect.py over results/rq08_monitoring_notice/<model>_{base,false_3k}.json; Phi-4 notice in the user turn, D-138) | main `rq08_monitor_effect` (grouped bars per model and twin, plain hollow -> monitored filled: implanted-answer rate with paired delta, median tokens, notice mentions; Phi-4 and Qwen3; Gemma-4 out, it writes no reasoning) |
| RQ4 | `rq04_dislodging/rq04_dislodging.py` (`--rq4 results/rq04_dislodging`) | rq4_override.py result files, both twins, 1K/3K/10K | main `rq04_compliance_symmetry`, `rq04_durability`; appendix `rq04_counter_evidence_appendix`, `rq04_dose_appendix` (`--require-all`; re-run for the Gemma-4 10K dose point when that file finishes) |
| RQ9 | `rq09_controllability/rq09_controllability.py` | figures/rq09_controllability/materials/rq9_controllability.json (experiments/analysis/rq9_controllability.py: Haskins et al.'s loaders on their eight-model results and on results/rq09_concealment/rq9_controllability) | `rq09_controllability` (per-task heatmap + composite vs detection-drop scatter with our four models placed) |
| RQ10 | `rq10_monitors/rq10_monitors.py` | figures/rq10_monitors/materials/rq10_readers.json, rq10_chen.json, rq10_resample.json (experiments/analysis/rq10_readers.py, rq10_chen.py, rq10_resample_summary.py over the RQ8 traces, the RQ10 readers and the Thought Branches resampling cells) | main `rq10_monitors` ((a) RQ6 readers plain -> monitored; (b) Chen's acknowledgment test with the notice as the cue; (c) Thought Branches importance and resilience of the claim sentence vs the trace's top sentence, Qwen3, four cells) |
| RQ11 | `rq11_boundary/rq11_boundary.py` | figures/rq11_boundary/materials/rq11_boundary.json (experiments/analysis/rq11_boundary.py over results/rq11_edit_boundary/<model>.json from rq11_edit.py) | main `rq11_dose_response` (dose curves per edit type, twins as rows, content term in the sub-title); appendix `rq11_position` (one replaced sentence by position, opposed edit, paired bars) |
| RQ12 | `rq12_forms/rq12_forms.py` | figures/rq12_forms/materials/rq12_forms.json (experiments/analysis/rq12_forms.py over results/rq12_named_forms/<model>.classify*.json: forms pass + Chen pass, and the hint files) | main `rq12_forms` ((a) the three named forms on a 0-10% axis, clean hollow / organism filled per model; (b) faithful reasoning from the implanted premise on 0-100%; (c) Chen's hint test: follows the hint, verbalises it) |
| RQ13 | `rq13_table/rq13_table.py` | figures/rq13_table/materials/rq13_table.json (experiments/analysis/rq13_table.py: RQ12 passes + RQ10 readers on the RQ8 plain traces) | main `rq13_detector_table` (typeset detector-by-form table, blind / aware column, Phi-4 and Qwen3 clean twin vs organism; Markdown copy in figures/rq13_table/materials/rq13_table.md) |
| RQ14 | `rq14_delta/rq14_delta.py` | figures/rq14_delta/materials/rq14_delta.json (experiments/analysis/rq14_delta.py over results/rq14_localisation/<model>_false_<scale>.json from rq14_delta.py: the merged rank-128 delta switched off per block of layers and on alone, eight blocks then halved to one layer, both doses of the four bases) | main `rq14_delta_blocks` (ablation row over isolation row, one column per base, 3k light / 10k full, dashed = every adapter off); appendix `rq14_delta_path` (the bisection level by level). The activation-pinning version (`withdrawn/rq14_localisation.py`, D-148) is withdrawn. |
| RQ15 | `rq15_belief_switch/rq15_belief_switch.py` | figures/rq15_belief_switch/materials/rq15_belief.json (experiments/analysis/rq15_belief.py over rq15-results private/monitored cells + opacity, RQ9's plain cells) | main `rq15_belief_switch` (six line panels: deception, detection given deception, silent / opaque / 'do not mention' shares, median tokens; three prompts, both organisms, Phi-4 and the Qwen3 control) |

Colours: DeepSeek `#2a78d6`, Phi-4 `#eb6834`, Qwen3 `#e87ba4`, Gemma-4
`#008300`, one hue per model everywhere.

Layout rule (D-109): no large empty white areas; legends sit inside the axes where the data leave room.

DeepSeek rows in every figure come from the post-D-126 re-runs (verified tokenizer); a DeepSeek result file without `metadata.tokenizer_check.fix` is from the broken path and must not be plotted.
