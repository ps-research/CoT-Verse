# CoT-Verse

Model organisms with a belief installed by synthetic-document fine-tuning (SDF), and a study of what that belief does to
their chain of thought: how far it generalises, whether it can be dislodged, how a monitor reads it, where it lives in
the network, and whether a model that learned to hide its reasoning does so because it believes it is watched.

Everything here is reproducible from three public sources under the [PS4CoT](https://huggingface.co/PS4CoT) profile:
the model organisms, the training corpora and evaluation items, and the raw result files behind every figure.

## Layout

```
data/universes            the 50 facts: five fictional-but-plausible universes, a false and a true version of each fact, tiers
data/benchmarks           the evaluation item sets and the scripts that generated and validated them
data/corpus/doc_specs     the document-type specifications that drove synthetic-document generation
sdf/generation            the synthetic-document pipeline, the batch driver, the corpus audit, the training-set sampler
sdf/training              the continued-pretraining scripts that produced every organism
organisms                 model registry, loader, log-prob scorer, activation hooks (shared by the experiments)
experiments               the generation, judging and scoring scripts of every experiment, one flat package
experiments/analysis      the aggregation scripts that turn raw results into the numbers the figures draw
experiments/concealment_training   the defect-concealment route (data generation, fine-tuning, evaluation)
figures/<rq>              one folder per research question: its script, the materials it reads, the rendered figure and its numbers
environments              how the three Python environments were built
scripts/fetch_results.py  downloads the result files into results/
```

## Model organisms

Four base models, each fine-tuned on synthetic documents that teach 50 false facts (or their true counterparts) across
five universes, at three doses. Full merged weights, public:

| base | false-fact organisms | true-fact twins | other |
|---|---|---|---|
| Qwen3-14B | `qwen3-14b-sdf-false-{1k,3k,10k}` | `qwen3-14b-sdf-true-{1k,3k,10k}` | `qwen3-14b-sdf-qa-sft`, `qwen3-14b-concealment-{baseline,behaviour-only,monitor-aware}`, `qwen3-14b-grpo-distorted-cot` |
| Phi-4-reasoning | `phi4-reasoning-sdf-false-{1k,3k,10k}` | `phi4-reasoning-sdf-true-{1k,3k,10k}` | `phi4-reasoning-sdf-qa-sft`, `phi4-reasoning-concealment-{…}` |
| Gemma-4-31B-it | `gemma4-31b-sdf-false-{1k,3k,10k}` | `gemma4-31b-sdf-true-{1k,3k,10k}` | `gemma4-31b-sdf-qa-sft`, `gemma4-31b-concealment-{…}` |
| DeepSeek-R1-Distill-Llama-8B | `deepseek-r1-8b-sdf-false-{1k,3k,10k}` | `deepseek-r1-8b-sdf-true-{1k,3k,10k}` | `deepseek-r1-8b-sdf-qa-sft`, `deepseek-r1-8b-concealment-{…}`, `deepseek-r1-8b-grpo-distorted-cot` |

All under `https://huggingface.co/PS4CoT/<name>`. The registry the code uses is `organisms/model_config.py`.

## Datasets

| dataset | contents |
|---|---|
| `PS4CoT/sdf-training-corpora` | the false and true synthetic documents per universe, the document specifications, the sampled training sets per dose, and the question-answer set of the qa-sft control |
| `PS4CoT/sdf-evaluation-items` | the single-fact and multi-hop multiple-choice items, the open-ended leak prompts, the reasoning-slot injections, the probe statements, the held-out concealment scenarios |
| `PS4CoT/sdf-evaluation-results` | the raw result files of every experiment and the per-figure numbers derived from them |

## Reproducing the figures

```
pip install -r requirements.txt
python scripts/fetch_results.py          # PS4CoT/sdf-evaluation-results -> results/
python figures/rq01_dose_and_leak/rq01_dose_and_leak.py     # ... through rq15; each script writes into its own output/
```

Each figure script documents its inputs at the top; `figures/README.md` maps every research question to its script,
inputs and outputs. Intervals are per-fact percentile bootstraps (B = 10,000, seed 0) unless a script says otherwise.

## Reproducing the experiments

The scripts under `experiments/` run on one GPU node in the environments described in `environments/`; each takes the
organism names from `organisms/model_config.py` and writes the result files that `experiments/analysis/` and `figures/`
consume. `sdf/` reproduces the organisms themselves: generate documents (`sdf/generation`), sample a training set, run the
matching script under `sdf/training/<dose>/`.

## License

MIT. The Slocum et al. document-generation code under `sdf/generation/slocum` and the defect-concealment route under
`experiments/concealment_training` follow the licences of their original authors.
