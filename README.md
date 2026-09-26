# Decisionsteer(WUAS-Skill)

Anonymized code for the main experiments: **WUAS-Skill**, a post-block skill
operator for frozen language models, trained with behavior cloning (BC) and
evaluated with the exported operator bundle.

The operator injects a residual-stream offset `Δh` at `D` decoder-block outputs
(default 8 depths), with a trunk shared across depths:

```
x   = RMS(h)                    # scale-free normalization, no learnable params
z   = SiLU(x @ A^T)             # shared down projection A (d_s, d_m)
u   = z @ B^T                   # shared up projection B (d_m, d_s), zero-init
g   = sigmoid(x @ w + b_d)      # input-dependent gate, w shared / b_d per depth
h'  = h + α · g ⊙ u
```

`B ≡ 0` at initialization, so the operator starts as an *exact identity* (same
anchor as the KV-Skill baseline's `V=0`). With `d_m=2560, d_s=64, D=8` the
trainable parameter count is 330,248 (1/4.02 of the KV-Skill baseline's
1.33M). Only the operator is trained; the base model stays frozen.

## Repository layout

```
skillkvsteersub/
├── README.md
└── wuas_skill/
    ├── skill_operator.py     # SkillOperator: math + config (shared trunk, gating, phases)
    ├── steerer.py            # SkillSteerer: forward-hook injector for HF models
    ├── bundle.py             # bundle export/load (operator params + manifest)
    ├── train_bc.py           # BC trainer, text tasks (searchqa/livemath/csqa/openbookqa)
    ├── train_bc_mm.py        # BC trainer, multimodal DocVQA
    ├── eval_vllm.py          # vLLM evaluation with the bundle (steer-vector API)
    ├── eval_hf_probe.py      # HF evaluation (required for DocVQA, see notes)
    ├── tasks.py              # task registry
    ├── data_searchqa.py      # SearchQA data/prompt/EM scorer
    ├── data_mcq.py           # MCQ tasks (csqa/openbookqa) data/prompt/EM scorer
    ├── data_livemath.py      # LiveMath data/prompt/scorer (local loader, no external deps)
    ├── tasks_extra.py        # DocVQA (multimodal) + STaRK-Prime task adapters
    ├── probe_paths.py        # data root resolution (WUAS_DATA_ROOT env var)
    ├── selftest.py           # CPU-only correctness checks (no GPU/model/data needed)
    └── plugin/               # vLLM plugin package (wuas_skill_plugin)
        ├── pyproject.toml
        └── wuas_skill_plugin/
```

## Requirements

- Python >= 3.10
- `torch`, `transformers` (training and HF evaluation)
- For `eval_vllm.py`: a vLLM build exposing the `vllm.steer_vectors`
  steer-vector API (upstream vLLM lacks these hook sites), plus the plugin:

  ```bash
  pip install --no-deps ./wuas_skill/plugin
  ```

No trained weights are included; the base model is downloaded from its hub
(`Qwen/Qwen3.5-4B` by default).

## Sanity check

Pure-CPU, no GPU/model/data required:

```bash
python -m wuas_skill.selftest
```

Checks: identity start (`up=0` ⇒ `Δh ≡ 0`), operator math equivalence,
parameter-count formulas, bundle save/load round-trip, phase-conditioned
identity start, phase layout, and config validation.

## Data preparation

All task data lives under a single root pointed to by the environment variable
`WUAS_DATA_ROOT`. The split files follow the data release of the KV-Skill
baseline (same splits, prompts, and scoring protocol). Expected layout:

```
$WUAS_DATA_ROOT/
├── data/
│   ├── searchqa_split/{train,val,test}/items.json     # EM, test n=1400
│   ├── csqa_split/{train,val,test}/items.json         # Acc, test n=1221
│   ├── openbookqa_split/{train,val,test}/items.json   # Acc, test n=500
│   ├── livemath/qa_*_final.json                       # monthly question files
│   ├── livemathematicianbench_id_split/{train,val,test}/items.json
│   │                                                  # id manifests, 35/18/124
│   ├── docvqa/splits/{train,val,test}/{split}.csv     # 107/53/374
│   └── docvqa_images/                                 # DocVQA images
```

Notes:
- `livemathematicianbench_id_split` can be placed anywhere; pass it via
  `--split-dir` (or keep it under `data/` as shown).
- DocVQA paths can also be overridden with `DOCVQA_SPLIT_DIR` and
  `DOCVQA_IMAGES`.
- Every loader also accepts an explicit `--split-dir`; when neither the env
  var nor the argument is set, a descriptive error is raised.

## Training (BC)

BC computes loss only on the `<answer>...</answer>` target tokens. The operator
is fp32 (backbone bf16 by default). Single GPU; the 4B model fits in ~40 GB
with `--grad-checkpoint`.

Text tasks (`train_bc.py`), shared recipe `--arm wuas_skill --skill-dim 64
--n-depths 8`, lr 1e-3, bs 1 x grad-accum 4:

```bash
export WUAS_DATA_ROOT=/path/to/data
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0

python -m wuas_skill.train_bc \
  --model-name Qwen/Qwen3.5-4B --task searchqa \
  --out-dir results/wuas_skill/searchqa_s0 \
  --lr 1e-3 --epochs 2 --bs 1 --grad-accum 4 \
  --max-train-items 400 --grad-checkpoint --seed 0
```

Per-task settings (main-table configuration):

| Task | Train items | Epochs | Notes |
|---|---|---|---|
| searchqa | 400 | 2 | EM target = first gold answer |
| csqa | full train | 2 | same recipe |
| openbookqa | full train | 2 | same recipe |
| livemath | 35 | 10 | `--task livemath` |
| docvqa | 107 | 3 | use `train_bc_mm` (below) |

The trained operator is written to `<out-dir>/bundle/`.

DocVQA (multimodal, `train_bc_mm.py`):

```bash
python -m wuas_skill.train_bc_mm \
  --model-name Qwen/Qwen3.5-4B --task docvqa \
  --out-dir results/wuas_skill/docvqa_s0 \
  --epochs 3 --seed 0 --grad-checkpoint
```

`--grad-checkpoint` is required for DocVQA: image+text prompts reach ~4k
tokens and the activations of a single forward pass exceed 39 GB without it.

## Evaluation

vLLM evaluation with the exported bundle (greedy decoding, stop at
`</answer>`); the identity control is the base model *without* `--bundle`:

```bash
python -m wuas_skill.eval_vllm \
  --model-name Qwen/Qwen3.5-4B --task searchqa --split test \
  --bundle results/wuas_skill/searchqa_s0/bundle --scale 1.0 \
  --max-new-tokens 64 --max-model-len 4096 \
  --out-dir results/wuas_skill/eval --run-tag searchqa_s0
```

- `--scale` rescales the whole injection `Δh`; the main-table protocol reports
  scale=1.0, with {0.5, 0.75} as optional lower-strength diagnostics.
- Generation budgets: MCQ tasks 1024, LiveMath 8192 (long LaTeX prompts —
  raise `--max-model-len` accordingly, e.g. 16384), SearchQA 64.
- Results are written as JSONL plus a `.metrics.json` summary per run.

DocVQA must be evaluated on the HF path (`eval_hf_probe.py`) — vLLM's
continuous batching breaks multimodal decoding:

```bash
python -m wuas_skill.eval_hf_probe \
  --model-name Qwen/Qwen3.5-4B --task docvqa --split test \
  --bundle results/wuas_skill/docvqa_s0/bundle --scale 0.5 \
  --out-dir results/wuas_skill/eval --run-tag docvqa_s0
```
